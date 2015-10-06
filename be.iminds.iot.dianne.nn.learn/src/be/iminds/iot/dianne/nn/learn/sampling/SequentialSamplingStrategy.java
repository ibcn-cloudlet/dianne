package be.iminds.iot.dianne.nn.learn.sampling;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;

public class SequentialSamplingStrategy implements SamplingStrategy{

	private int index = 0;
	private Dataset dataset;
	
	public SequentialSamplingStrategy(Dataset d) {
		this.dataset = d;
	}
	
	@Override
	public int next() {
		if(index >= dataset.size()){
			index = 0;
		}
		return index++;
	}

}
