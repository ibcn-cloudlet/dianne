package be.iminds.iot.dianne.nn.learn.sampling;

import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;

public class SequentialSamplingStrategy implements SamplingStrategy{

	private int index;

	private int[] indices;
	
	public SequentialSamplingStrategy(int[] indices) {
		this.indices = indices;
		this.index = 0;
	}
	
	@Override
	public int next() {
		if(index >= indices.length){
			index = 0;
		}
		return indices[index++];
	}

}
