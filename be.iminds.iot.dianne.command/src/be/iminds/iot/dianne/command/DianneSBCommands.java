package be.iminds.iot.dianne.command;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.apache.felix.service.command.Descriptor;
import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.dataset.DianneDatasets;
import be.iminds.iot.dianne.api.dataset.Sequence;
import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=entropy"},
		immediate=true)
public class DianneSBCommands {
	
	static Random random = new Random();
	
	BundleContext context;

	Dianne dianne;
	DiannePlatform platform;
	DianneDatasets datasets;
	
	@Descriptor("Trace the entropy for the SB model.")
	public void entropy(
			@Descriptor("prior model")
			String priorName, 
			@Descriptor("posterior model")
			String posteriorName,
			@Descriptor("likelihood model")
			String likelihoodName,
			@Descriptor("encoder model")
			String encoderName,
			@Descriptor("parameter tag")
			String tag,
			@Descriptor("dataset")
			String dataset,
			@Descriptor("number of states")
			int noStates,
			@Descriptor("number of sequences")
			int noSequences,
			@Descriptor("number of joint steps")
			int noJoint,
			@Descriptor("number of separate steps")
			int noSeparate,
			@Descriptor("number of samples")
			int noSamples
		) {
		try {
			NeuralNetworkInstanceDTO priorDTO = platform.deployNeuralNetwork(priorName, new String[]{tag});
			NeuralNetworkInstanceDTO posteriorDTO = platform.deployNeuralNetwork(posteriorName, new String[]{tag});
			NeuralNetworkInstanceDTO likelihoodDTO = platform.deployNeuralNetwork(likelihoodName, new String[]{tag});
			NeuralNetworkInstanceDTO encoderDTO = platform.deployNeuralNetwork(encoderName, new String[]{tag});
			
			NeuralNetwork prior = dianne.getNeuralNetwork(priorDTO).getValue();
			NeuralNetwork posterior = dianne.getNeuralNetwork(posteriorDTO).getValue();
			NeuralNetwork likelihood = dianne.getNeuralNetwork(likelihoodDTO).getValue();
			NeuralNetwork encoder = dianne.getNeuralNetwork(encoderDTO).getValue();
			
			UUID[] priorIn = prior.getModuleIds("State","Action");
			UUID[] priorOut = prior.getModuleIds("Output");
			UUID[] posteriorIn = posterior.getModuleIds("State","Action","Observation");
			UUID[] posteriorOut = posterior.getModuleIds("Output");
			
			if(!datasets.isExperiencePool(dataset))
				throw new InstantiationException("Dataset is not an ExperiencePool");
			ExperiencePool pool = (ExperiencePool) datasets.getDataset(dataset);
			
			Tensor priorState = new Tensor(noSamples, noStates);
			Tensor posteriorState = new Tensor(noSamples, noStates);
			Tensor action = new Tensor(noSamples, pool.actionDims());
			Tensor observation = new Tensor(noSamples, pool.stateDims());
			
			Tensor prior2prior = new Tensor(noStates);
			Tensor prior2posterior = new Tensor(noStates);
			Tensor posterior2prior = new Tensor(noStates);
			Tensor posterior2posterior = new Tensor(noStates);
			
			double[][] priorEntropy = new double[noSeparate][noSequences];
			double[][] posteriorEntropy = new double[noSeparate][noSequences];
			double[][] forwardDivergence = new double[noSeparate][noSequences];
			double[][] backwardDivergence = new double[noSeparate][noSequences];
			double[][] priorLikelihood = new double[noSeparate][noSequences];
			double[][] posteriorLikelihood = new double[noSeparate][noSequences];
			
			for(int s = 0; s < noSequences; s++) {
				System.out.print(s + ": ");
				Sequence<ExperiencePoolSample> sequence = pool.getSequence(s);
				
				posteriorState.fill(0.f);
				action.fill(0.f);
				
				for(int t = 0; t < noJoint && t < sequence.size(); t++) {
					ExperiencePoolSample sample = sequence.get(t);
					expand(observation, sequence.get(t).getState(), noSamples);
					
					Tensor posteriorDistribution = posterior.forward(posteriorIn, posteriorOut, new Tensor[]{posteriorState, action, observation}).getValue().tensor;
//					Tensor posteriorDistribution = new Tensor(1, noStates*2);
//					posteriorDistribution.narrow(1, 0, noStates).fill(0.f);
//					posteriorDistribution.narrow(1, noStates, noStates).fill(1.f);
					sampleFromGaussianMixture(posteriorState, posteriorDistribution, noSamples);
					
					expand(action, sample.getAction(), noSamples);
					System.out.print('-');
				}
				
				posteriorState.copyInto(priorState);
				
				for(int t = noJoint; t < noJoint + noSeparate && t < sequence.size(); t++) {
					ExperiencePoolSample sample = sequence.get(t);
					expand(observation, sequence.get(t).getState(), noSamples);
					
					Tensor priorDistribution = prior.forward(priorIn, priorOut, new Tensor[]{priorState, action}).getValue().tensor;
//					Tensor priorDistribution = new Tensor(1, noStates*2);
//					priorDistribution.narrow(1, 0, noStates).fill(0.f);
//					priorDistribution.narrow(1, noStates, noStates).fill(1.f);
					sampleFromGaussianMixture(priorState, priorDistribution, noSamples);
					
					Tensor posteriorDistribution = posterior.forward(posteriorIn, posteriorOut, new Tensor[]{posteriorState, action, observation}).getValue().tensor;
//					Tensor posteriorDistribution = new Tensor(1, noStates*2);
//					posteriorDistribution.narrow(1, 0, noStates).fill(0.f);
//					posteriorDistribution.narrow(1, noStates, noStates).fill(1.f);
					sampleFromGaussianMixture(posteriorState, posteriorDistribution, noSamples);
					
					TensorOps.log(prior2prior, gaussianMixtureLikelihood(priorState, priorDistribution));
					TensorOps.log(prior2posterior, gaussianMixtureLikelihood(priorState, posteriorDistribution));
					TensorOps.log(posterior2prior, gaussianMixtureLikelihood(posteriorState, priorDistribution));
					TensorOps.log(posterior2posterior, gaussianMixtureLikelihood(posteriorState, posteriorDistribution));
					
					priorEntropy[t - noJoint][s] = -TensorOps.mean(prior2prior);
					posteriorEntropy[t - noJoint][s] = -TensorOps.mean(posterior2posterior);
					forwardDivergence[t - noJoint][s] = TensorOps.mean(TensorOps.sub(null, prior2prior, prior2posterior));
					backwardDivergence[t - noJoint][s] = TensorOps.mean(TensorOps.sub(null, posterior2posterior, posterior2prior));
					
					Tensor features = encoder.forward(sequence.get(t).getState());
					features.reshape(1, features.size());
					
					priorDistribution = likelihood.forward(priorState).clone();
					posteriorDistribution = likelihood.forward(posteriorState).clone();
					
					priorLikelihood[t - noJoint][s] = Math.log(gaussianMixtureLikelihood(features, priorDistribution).get(0));
					posteriorLikelihood[t - noJoint][s] = Math.log(gaussianMixtureLikelihood(features, posteriorDistribution).get(0));
					
					expand(action, sample.getAction(), noSamples);
					System.out.print('*');
				}
				System.out.println();
			}
			
			Function<double[][], List<Double>> average = data -> Arrays.stream(data).mapToDouble(r -> Arrays.stream(r).average().getAsDouble()).boxed().collect(Collectors.toList());
			
			System.out.println("Prior entropy:\n" + average.apply(priorEntropy));
			System.out.println("Posterior entropy:\n" + average.apply(posteriorEntropy));
			System.out.println("Forward divergence:\n" + average.apply(forwardDivergence));
			System.out.println("Backward divergence:\n" + average.apply(backwardDivergence));
			System.out.println("Prior likelihood:\t" + average.apply(priorLikelihood));
			System.out.println("Posterior likelihood:\t" + average.apply(posteriorLikelihood));
			
			platform.undeployNeuralNetwork(priorDTO);
			platform.undeployNeuralNetwork(posteriorDTO);
			platform.undeployNeuralNetwork(likelihoodDTO);
			platform.undeployNeuralNetwork(encoderDTO);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private static Tensor gaussianMixtureLikelihood(Tensor samples, Tensor distribution) {
		int noSamples = samples.size(0);
		int noStates = samples.size(1);
		int noComponents = distribution.size(0);
		
		Tensor result = new Tensor(noSamples);
		Tensor sample = new Tensor(noComponents, noStates);
		
		Tensor mean = distribution.narrow(1, 0, noStates);
		Tensor stdev = distribution.narrow(1, noStates, noStates);
		Tensor stdevSq = TensorOps.cmul(null, stdev, stdev);
		TensorOps.mul(stdevSq, stdevSq, 2.f);
		Tensor scale = TensorOps.mul(null, stdevSq, (float) Math.PI);
		TensorOps.log(scale, scale);
		
		for(int s = 0; s < noSamples; s++) {
			expand(sample, samples.select(0, s), noComponents);
			
			//var y = Math.exp(-(x-mu)*(x-mu)/(2*sigma*sigma))/Math.sqrt(2*sigma*sigma*Math.PI);
			//var log_y = -(x-mu)*(x-mu)/(s*sigma*sigma)-Math.log(2*sigma*sigma*Math.PI)/2
			TensorOps.sub(sample, sample, mean);
			TensorOps.cmul(sample, sample, sample);
			TensorOps.cdiv(sample, sample, stdevSq);
			TensorOps.mul(sample, sample, -1.f);
			TensorOps.sub(sample, sample, 0.5f, scale);
			
			Tensor temp = sample.select(1, 0);
			for(int i = 1; i < noStates; i++)
				TensorOps.add(temp, temp, sample.select(1, i));
			
			TensorOps.exp(temp, temp);
			result.set(TensorOps.mean(temp), s);
		}
		return result;
	}

	private static Tensor expand(Tensor batch, Tensor value, int batchSize) {
		for(int i = 0; i < batchSize; i++)
			value.copyInto(batch.select(0, i));
		return batch;
	}
	
	private static Tensor sampleFromGaussianMixture(Tensor result, Tensor distribution, int batchSize) {
		for(int i = 0; i < batchSize; i++)
			sampleFromGaussianMixture(result.select(0, i), distribution);
		return result;
	}
	
	private static Tensor sampleFromGaussianMixture(Tensor result, Tensor distribution) {
		return sampleFromGaussian(result, distribution.select(0, random.nextInt(distribution.size(0))));
	}
	
	private static Tensor sampleFromGaussian(Tensor result, Tensor distribution) {
		int size = distribution.size()/2;
		Tensor means = distribution.narrow(0, 0, size);
		Tensor stdevs = distribution.narrow(0, size, size);
		
		Tensor random = new Tensor(means.size());
		random.randn();
		
		TensorOps.cmul(result, random, stdevs);
		TensorOps.add(result, result, means);
		return result;
	}
	
	@Reference
	void setDiannePlatform(DiannePlatform p){
		platform = p;
	}
	
	@Reference
	void setDianne(Dianne d){
		dianne = d;
	}

	@Reference
	void setDianneDatasets(DianneDatasets d){
		datasets = d;
	}
	
}
