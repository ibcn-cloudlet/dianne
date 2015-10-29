package be.iminds.iot.dianne.nn.learn.sampling;

import java.util.Random;

import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;

public class RandomSamplingStrategy implements SamplingStrategy{

	private Random random = new Random(System.currentTimeMillis());
	
	private int[] indices;
	
	public RandomSamplingStrategy(int[] indices) {
		this.indices = indices;
	}
	
	@Override
	public int next() {
		return indices[random.nextInt(indices.length)];
	}

}
