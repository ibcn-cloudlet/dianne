package be.iminds.iot.dianne.nn.train.eval;

import be.iminds.iot.dianne.nn.module.io.Input;
import be.iminds.iot.dianne.nn.module.io.Output;
import be.iminds.iot.dianne.nn.train.Dataset;
import be.iminds.iot.dianne.nn.train.Evaluation;
import be.iminds.iot.dianne.nn.train.Evaluator;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class ArgMaxEvaluator implements Evaluator {

	protected static final TensorFactory factory = TensorFactory.getFactory(TensorFactory.TensorType.JAVA);
	
	@Override
	public Evaluation evaluate(Input input, Output output, Dataset data) {
		Tensor confusion = factory.createTensor(data.outputSize(), data.outputSize());
		confusion.fill(0.0f);
		
		for(int i=0;i<data.size();i++){

			// Read samples from dataset
			Tensor in = data.getInputSample(i);
			Tensor out = data.getOutputSample(i);
			
			// Forward through input module
			input.forward(input.getId(), in);
	
			int predicted = factory.getTensorMath().argmax(output.getOutput());
			int real = factory.getTensorMath().argmax(out);
			
			confusion.set(confusion.get(real, predicted)+1, real, predicted);
		}
		
		return new Evaluation(confusion);
	}

}
