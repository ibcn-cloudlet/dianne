package test;

import java.util.Map;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.LearningStrategy;

public class TestLearningStrategy implements LearningStrategy {

	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		System.out.println("SETUP");
	}

	@Override
	public LearnProgress processIteration(long i) throws Exception {
		System.out.println("PROCESS "+i);
		return new LearnProgress(i, 0.0f);
	}

}
