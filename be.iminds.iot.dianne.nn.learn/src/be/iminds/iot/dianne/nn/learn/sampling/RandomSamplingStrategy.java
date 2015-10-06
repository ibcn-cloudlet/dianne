package be.iminds.iot.dianne.nn.learn.sampling;

import java.util.Random;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;

public class RandomSamplingStrategy implements SamplingStrategy{

	private Random random = new Random(System.currentTimeMillis());
	
	private Dataset dataset;
	
	public RandomSamplingStrategy(Dataset d) {
		this.dataset = d;
	}
	
	@Override
	public int next() {
		return random.nextInt(dataset.size());
	}

}
