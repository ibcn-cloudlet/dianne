package be.iminds.iot.dianne.nn.learn.sampling;

import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;

public class SequentialSamplingStrategy implements SamplingStrategy{

	private int index;

	private int startIndex;
	private int endIndex;
	
	public SequentialSamplingStrategy(int startIndex, int endIndex) {
		this.startIndex = startIndex;
		this.endIndex = endIndex;
		this.index = startIndex;
	}
	
	@Override
	public int next() {
		if(index >= endIndex){
			index = startIndex;
		}
		return index++;
	}

}
