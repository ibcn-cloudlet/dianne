package be.iminds.iot.dianne.nn.learn.processors;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class StochasticGradientDescentProcessor extends AbstractProcessor {

	// dataset
	protected final Dataset dataset;
	// sample strategy
	protected final SamplingStrategy sampling;
	// error criterion
	protected final Criterion criterion;
	// learning rate
	protected final float learningRate;
	// batch size
	protected final int batchSize;
	
	// current error
	protected float error = 0;
	
	public StochasticGradientDescentProcessor(TensorFactory factory, 
			NeuralNetwork nn, 
			DataLogger logger, 
			Dataset d, 
			SamplingStrategy s,
			Criterion c,
			float learningRate,
			int batchSize) {
		super(factory, nn, logger);
	
		this.dataset = d;
		this.sampling = s;
		this.criterion = c;
		
		this.learningRate = learningRate;
		this.batchSize = batchSize;
	}
	
	
	@Override
	public float processNext() {
		error = 0;

		for(int i=0;i<batchSize;i++){
			// new sample
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
