package be.iminds.iot.dianne.rl.learn.factory;

import java.util.Map;

import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.Processor;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.api.rl.ExperiencePool;
import be.iminds.iot.dianne.nn.learn.factory.LearnerFactory;
import be.iminds.iot.dianne.nn.learn.processors.AbstractProcessor;
import be.iminds.iot.dianne.rl.learn.processors.TimeDifferenceProcessor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class QLearnerFactory {

	public static Processor createProcessor(
		TensorFactory factory, 
		NeuralNetwork nn, 
		NeuralNetwork target,
		ExperiencePool pool, 
		Map<String, String> config,
		DataLogger logger){

		AbstractProcessor p = createTDProcessor(factory, nn, target, pool, config, logger);
		p = LearnerFactory.addRegularization(p, config);
		p = LearnerFactory.addMomentum(p, config);
		return (Processor) p;
	}
	
	public static AbstractProcessor createTDProcessor(TensorFactory factory, 
			NeuralNetwork nn, NeuralNetwork target,
			ExperiencePool pool, Map<String, String> config, DataLogger logger){
		
		float learningRate = 0.001f;
		if(config.get("learningRate")!=null){
			learningRate = Float.parseFloat(config.get("learningRate"));
		}
		
		int batchSize = 10;
		if(config.get("batchSize")!=null){
			batchSize = Integer.parseInt(config.get("batchSize"));
		}
		
		float discount = 0.99f;
		if(config.containsKey("discount"))
			discount = Float.parseFloat(config.get("discount"));
		
		Criterion c = LearnerFactory.createCriterion(factory, config);
		SamplingStrategy s = LearnerFactory.createSamplingStrategy(pool, config);
		
		AbstractProcessor p = new TimeDifferenceProcessor(factory, nn, target, logger, pool, s, c, 
				learningRate, batchSize, discount);
		
		System.out.println("TimeDifferenceRL");
		System.out.println("* criterion = "+c.getClass().getName());
		System.out.println("* sampling = "+ s.getClass().getName());
		System.out.println("* learningRate = "+learningRate);
		System.out.println("* batchSize = "+batchSize);
		System.out.println("* discount = "+discount);
		System.out.println("---");
		
		return p;
	}
}
