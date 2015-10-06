package be.iminds.iot.dianne.nn.learn.processors;

import java.util.Map;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.nn.learn.criterion.MSECriterion;
import be.iminds.iot.dianne.nn.learn.criterion.NLLCriterion;
import be.iminds.iot.dianne.nn.learn.sampling.RandomSamplingStrategy;
import be.iminds.iot.dianne.nn.learn.sampling.SequentialSamplingStrategy;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class StochasticGradientDescentProcessor extends AbstractProcessor {

	// sample strategy
	protected SamplingStrategy sampling;
	// error criterion
	protected Criterion criterion;
	// learning rate
	protected float learningRate = 0.01f;
	// batch size
	protected int batchSize = 10;
	
	// current error
	protected float error = 0;
	
	public StochasticGradientDescentProcessor(TensorFactory factory, 
			NeuralNetwork nn, 
			Dataset dataset, 
			Map<String, String> config,
			DataLogger logger) {
		super(factory, nn, dataset, config, logger);
	
		this.criterion = new MSECriterion(factory);
		String c = config.get("criterion");
		if(c!=null){
			if(c.equals("NLL")){
				criterion = new NLLCriterion(factory);
			} else if(c.equals("MSE")){
				criterion = new MSECriterion(factory);
			}
		}

		String l = config.get("learningRate");
		if(l!=null){
			learningRate = Float.parseFloat(l);
		}
		
		String b = config.get("batchSize");
		if(b!=null){
			batchSize = Integer.parseInt(b);
		}
		
		String s = config.get("samplingStrategy");
		if(s != null){
			if(s.equals("random")){
				sampling = new RandomSamplingStrategy(dataset);
			} else if(s.equals("sequential")){
				sampling = new SequentialSamplingStrategy(dataset);
			} else {
				// random is default
				sampling = new RandomSamplingStrategy(dataset);
			}
		} else {
			// random is default
			sampling = new RandomSamplingStrategy(dataset);
		}
				 
		
		System.out.println("StochasticGradientDescent");
		System.out.println("* criterion = "+criterion.getClass().getName());
		System.out.println("* learningRate = "+learningRate);
		System.out.println("* batchSize = "+batchSize);
		System.out.println("---");
	}
	
	
	@Override
	public float processNext() {
		error = 0;

		for(int i=0;i<batchSize;i++){
			int index = sampling.next();
			
			Tensor in = dataset.getInputSample(index);

			// forward
			Tensor out = nn.forward(in, ""+index);
			
			// evaluate criterion
			Tensor gradOut = getGradOut(out, index);
			
			// backward
			Tensor gradIn = nn.backward(gradOut, ""+index);
			
			// acc grad params
			accGradParameters();
		}

		// apply learning rate
		applyLearningRate();
		
		return error/batchSize;
	}

	protected void accGradParameters(){
		// acc gradParameters
		nn.getTrainables().values().stream().forEach(m -> m.accGradParameters());
	}
	
	protected void applyLearningRate(){
		// multiply with learning rate
		nn.getTrainables().values().stream().forEach(
				m -> factory.getTensorMath().mul(m.getDeltaParameters(), m.getDeltaParameters(), -learningRate));
	}
	
	protected Tensor getGradOut(Tensor out, int index){
		Tensor e = criterion.error(out, dataset.getOutputSample(index));
		error += e.get(0);
		return criterion.grad(out, dataset.getOutputSample(index));
	}
}
