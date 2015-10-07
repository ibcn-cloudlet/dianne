package be.iminds.iot.dianne.nn.learn.sampling;

import java.util.Random;

import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;

public class RandomSamplingStrategy implements SamplingStrategy{

	private Random random = new Random(System.currentTimeMillis());
	
	private int startIndex;
	private int endIndex;
	
	public RandomSamplingStrategy(int startIndex, int endIndex) {
		this.startIndex = startIndex;
		this.endIndex = endIndex;
	}
	
	@Override
	public int next() {
		return random.nextInt(endIndex-startIndex) + startIndex;
	}

}
