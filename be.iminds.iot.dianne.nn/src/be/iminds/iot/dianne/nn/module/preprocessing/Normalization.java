package be.iminds.iot.dianne.nn.module.preprocessing;

import java.util.UUID;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.api.nn.module.Preprocessor;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class Normalization extends AbstractModule implements Preprocessor {

	// TODO per channel normalization?
	private float mean = 0.0f;
	private float std = 1.0f;
	
	private boolean preprocessed = false;
	
	public Normalization(TensorFactory factory){
		super(factory);
	}
	
	public Normalization(TensorFactory factory, UUID id){
		super(factory, id);
	}
	
	public Normalization(TensorFactory factory, UUID id, float mean, float std){
		super(factory, id);
		this.mean = mean;
		this.std = std;
		preprocessed = true;
	}

	@Override
	protected void forward() {
		output = factory.getTensorMath().sub(output, input, mean);
		output = factory.getTensorMath().div(output, output, std);

	}

	@Override
	protected void backward() {
		// does not matter?
	}

	@Override
	public void preprocess(Dataset data) {
		int n = 0;
		float m = 0;
		float m2 = 0;
		float d,x;
		for(int i=0;i<data.size();i++){
			Tensor input = data.getInputSample(i);
			float[] inputData = input.get();
			// TODO normalize over all data, what if this is subtensor?
			for(int k=0;k<inputData.length;k++){
				x = inputData[k];
				n++;
				d = x - m;
				m = m + d/n;
				m2 = m2 + d*(x - m);
			}
		}
		
		this.mean = m;
		if(n < 2){
			this.std = 0;
		} else {
			std = (float) Math.sqrt(m2/(n-1));
		}
		
		preprocessed = true;
	}

	@Override
	public Tensor getParameters() {
		Tensor params = factory.createTensor(2);
		params.set(mean, 0);
		params.set(std, 1);
		return params;
	}

	@Override
	public void setParameters(Tensor params) {
		mean = params.get(0);
		std = params.get(1);
		
		preprocessed = true;
	}

	@Override
	public boolean isPreprocessed() {
		return preprocessed;
	}
}
