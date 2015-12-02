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
package be.iminds.iot.dianne.nn.learn.factory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.Processor;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.nn.learn.criterion.MSECriterion;
import be.iminds.iot.dianne.nn.learn.criterion.NLLCriterion;
import be.iminds.iot.dianne.nn.learn.processors.AbstractProcessor;
import be.iminds.iot.dianne.nn.learn.processors.MomentumProcessor;
import be.iminds.iot.dianne.nn.learn.processors.RegularizationProcessor;
import be.iminds.iot.dianne.nn.learn.processors.StochasticGradientDescentProcessor;
import be.iminds.iot.dianne.nn.learn.sampling.RandomSamplingStrategy;
import be.iminds.iot.dianne.nn.learn.sampling.SequentialSamplingStrategy;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class LearnerFactory {

	public static Processor createProcessor(TensorFactory factory, 
			NeuralNetwork nn, 
			Dataset d, 
			Map<String, String> config,
			DataLogger logger){
		AbstractProcessor p = createSGDProcessor(factory, nn, d, config, logger);
		p = addRegularization(p, config);
		p = addMomentum(p, config);
		return (Processor) p;
	}
	
	public static AbstractProcessor createSGDProcessor(TensorFactory factory, 
			NeuralNetwork nn, Dataset d, Map<String, String> config, DataLogger logger){
		float learningRate = 0.001f;
		if(config.get("learningRate")!=null){
			learningRate = Float.parseFloat(config.get("learningRate"));
		}
		
		int batchSize = 10;
		if(config.get("batchSize")!=null){
			batchSize = Integer.parseInt(config.get("batchSize"));
		}
		
		Criterion c = createCriterion(factory, config);
		SamplingStrategy s = createSamplingStrategy(d, config);
		
		AbstractProcessor p = new StochasticGradientDescentProcessor(factory, nn, logger,d, s, c, 
				learningRate, batchSize);
		
		System.out.println("StochasticGradientDescent");
		System.out.println("* criterion = "+c.getClass().getName());
		System.out.println("* sampling = "+ s.getClass().getName());
		System.out.println("* learningRate = "+learningRate);
		System.out.println("* batchSize = "+batchSize);
		System.out.println("---");
		
		return p;
	}
	
	public static AbstractProcessor addRegularization(AbstractProcessor p, Map<String, String> config){
		if(config.get("regularization")!=null){
			float regularization = Float.parseFloat(config.get("regularization"));
			RegularizationProcessor r = new RegularizationProcessor(p, regularization);
			
			System.out.println("Regularization");
			System.out.println("* factor = "+regularization);
			System.out.println("---");
			
			return r;
		} else {
			return p;
		}
	}
	
	public static AbstractProcessor addMomentum(AbstractProcessor p, Map<String, String> config){
		if(config.get("momentum")!=null){
			float momentum = Float.parseFloat(config.get("momentum"));
			AbstractProcessor m = new MomentumProcessor(p, momentum);
			
			System.out.println("Momentum");
			System.out.println("* rate = "+momentum);
			System.out.println("---");
			
			return m;
		} else {
			return p;
		}
		
	}
	
	public static Criterion createCriterion(TensorFactory factory, Map<String, String> config){
		Criterion criterion = new MSECriterion(factory);
		String c = config.get("criterion");
		if(c!=null){
			if(c.equals("NLL")){
				criterion = new NLLCriterion(factory);
			} else if(c.equals("MSE")){
				criterion = new MSECriterion(factory);
			}
		}
		return criterion;
	}
	
	public static SamplingStrategy createSamplingStrategy(Dataset d, Map<String, String> config){
		
		int[] indices = null;
		String range = config.get("range");
		if(range!=null){
			indices = parseRange(range);
			
			System.out.println("Dataset range");
			System.out.println("* range = "+range);
			System.out.println("---");
		} else {
			int startIndex = 0;
			int endIndex = d.size();
			
			String start = config.get("startIndex");
			if(start!=null){
				startIndex = Integer.parseInt(start);
			}
			
			String end = config.get("endIndex");
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

		SamplingStrategy sampling = null;
		String s = config.get("samplingStrategy");
		if(s != null){
			if(s.equals("random")){
				sampling = new RandomSamplingStrategy(indices);
			} else if(s.equals("sequential")){
				sampling = new SequentialSamplingStrategy(indices);
			} else {
				// random is default
				sampling = new RandomSamplingStrategy(indices);
			}
		} else {
			// random is default
			sampling = new RandomSamplingStrategy(indices);
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
