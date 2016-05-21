/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  iMinds - IBCN - UGent
 *
 * This file is part of DIANNE.
 *
 * DIANNE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.nn.learn;

import java.util.ArrayList;
import java.util.Map;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.nn.learn.criterion.MSECriterion;
import be.iminds.iot.dianne.nn.learn.criterion.NLLCriterion;
import be.iminds.iot.dianne.nn.learn.processors.AdadeltaProcessor;
import be.iminds.iot.dianne.nn.learn.processors.AdagradProcessor;
import be.iminds.iot.dianne.nn.learn.processors.MomentumProcessor;
import be.iminds.iot.dianne.nn.learn.processors.NesterovMomentumProcessor;
import be.iminds.iot.dianne.nn.learn.processors.RMSpropProcessor;
import be.iminds.iot.dianne.nn.learn.processors.RegularizationProcessor;
import be.iminds.iot.dianne.nn.learn.processors.StochasticGradientDescentProcessor;
import be.iminds.iot.dianne.nn.learn.sampling.RandomSamplingStrategy;
import be.iminds.iot.dianne.nn.learn.sampling.SequentialSamplingStrategy;

public class LearnerUtil {

	public static GradientProcessor createGradientProcessor(NeuralNetwork nn, Dataset d,
			Map<String, String> config,DataLogger logger){
		return addMomentum(addRegularization(createSGDProcessor(nn, d, config, logger), config), config);
	}
	
	public static GradientProcessor createSGDProcessor(NeuralNetwork nn, Dataset d,
			Map<String, String> config, DataLogger logger){
		String method = "SGD";
		if(config.containsKey("method"))
			method = config.get("method");
		
		float learningRate = 0.01f;
		if(config.containsKey("learningRate"))
			learningRate = Float.parseFloat(config.get("learningRate"));
		
		float decayRate = 0.9f;
		if(config.containsKey("decayRate"))
			decayRate = Float.parseFloat(config.get("decayRate"));
		
		GradientProcessor p = null;
		
		switch(method) {
		case "Adadelta":
			p = new AdadeltaProcessor(nn, logger, decayRate);
			
			System.out.println("Adadelta");
			System.out.println("* decayRate = "+decayRate);
			System.out.println("---");
			break;
		case "Adagrad":
			p = new AdagradProcessor(nn, logger, learningRate);
			
			System.out.println("Adagrad");
			System.out.println("* learningRate = "+learningRate);
			System.out.println("---");
			break;
		case "RMSprop":
			p = new RMSpropProcessor(nn, logger, learningRate, decayRate);
			
			System.out.println("RMSprop");
			System.out.println("* learningRate = "+learningRate);
			System.out.println("* decayRate = "+decayRate);
			System.out.println("---");
			break;	
		default:
			if(!method.equals("SGD"))
				System.out.println("Method "+method+" unknown, fall back to SGD");
			
			p = new StochasticGradientDescentProcessor(nn, logger, learningRate);
			
			System.out.println("StochasticGradientDescent");
			System.out.println("* learningRate = "+learningRate);
			System.out.println("---");
			break;
		}

		return p;
	}
	
	public static GradientProcessor addRegularization(GradientProcessor p, Map<String, String> config){
		if(config.containsKey("regularization")) {
			float regularization = Float.parseFloat(config.get("regularization"));
			
			if(regularization==0.0f)
				return p;
			
			RegularizationProcessor r = new RegularizationProcessor(p, regularization);
			
			System.out.println("Regularization");
			System.out.println("* factor = "+regularization);
			System.out.println("---");
			
			return r;
		} else {
			return p;
		}
	}
	
	public static GradientProcessor addMomentum(GradientProcessor p, Map<String, String> config){
		if(config.containsKey("momentum")) {
			float momentum = Float.parseFloat(config.get("momentum"));
			
			if(momentum==0.0f)
				return p;
			
			GradientProcessor m = new MomentumProcessor(p, momentum);
			
			System.out.println("Momentum");
			System.out.println("* rate = "+momentum);
			System.out.println("---");
			
			return m;
		} else if(config.containsKey("nesterov")) {
			float momentum = Float.parseFloat(config.get("nesterov"));
			
			if(momentum==0.0f)
				return p;
			
			GradientProcessor m = new NesterovMomentumProcessor(p, momentum);
			
			System.out.println("Nesterov momentum");
			System.out.println("* rate = "+momentum);
			System.out.println("---");
			
			return m;
		} else {
			return p;
		}
	}
	
	public static Criterion createCriterion(Map<String, String> config){
		String c = "MSE";
		if(config.containsKey("criterion"))
			c = config.get("criterion");
		
		Criterion criterion = null;
		
		switch(c) {
		case "NLL" :
			criterion = new NLLCriterion();
			break;
		default:
			if(!c.equals("MSE"))
				System.out.println("Criterion "+c+" unknown, fall back to MSE");
			
			criterion = new MSECriterion();
			break;
		}
		
		return criterion;
	}
	
	public static SamplingStrategy createSamplingStrategy(Dataset d, Map<String, String> config){
		
		int[] indices = null;
		String range = config.get("range");
		if(range!=null){
			indices = parseRange(range);
			
			System.out.println("Dataset range");
			if(range.contains(":"))
				System.out.println("* range = "+range);
			else 
				System.out.println("* "+indices.length+" indices selected");			System.out.println("---");
		} else  {
			String start = config.get("startIndex");
			String end = config.get("endIndex");
			
			if(start!=null || end !=null){
				int startIndex = 0;
				int endIndex = d.size();
				
				if(start!=null){
					startIndex = Integer.parseInt(start);
				}
				
				if(end!=null){
					endIndex = Integer.parseInt(end);
				}
				
				int index = startIndex;
				indices = new int[endIndex-startIndex];
				for(int i=0;i<indices.length;i++){
					indices[i] = index++;
				}
				
				System.out.println("Dataset range");
				System.out.println("* startIndex = "+startIndex);
				System.out.println("* endIndex = "+endIndex);
				System.out.println("---");
			}
		}

		String s = "random";
		if(config.containsKey("samplingStrategy"))
			s = config.get("samplingStrategy");
		
		SamplingStrategy sampling = null;

		switch(s) {
		case "sequential":
			sampling = new SequentialSamplingStrategy(d, indices);
			break;
		default:
			if(!s.equals("random"))
				System.out.println("Sampling strategy "+s+" unknown, fall back to random");
				
			sampling = new RandomSamplingStrategy(d, indices);
		}
		
		return sampling;
	}
	
	private static int[] parseRange(String range){
		ArrayList<Integer> list = new ArrayList<>();
		String[] subranges = range.split(",");
		for(String subrange : subranges){
			String[] s = subrange.split(":");
			if(s.length==2){
				for(int i=Integer.parseInt(s[0]);i<Integer.parseInt(s[1]);i++){
					list.add(i);
				}
			} else {
				list.add(Integer.parseInt(s[0]));
			}
		}
		int[] array = new int[list.size()];
		for(int i=0;i<list.size();i++){
			array[i] = list.get(i);
		}
		return array;
	}
}
