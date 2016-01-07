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
package be.iminds.iot.dianne.rnn.command;

import java.util.HashMap;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

/**
 * Separate component for learn commands ... should be moved to the command bundle later on
 */
@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=generate",
				  "osgi.command.function=bptt",
				  "osgi.command.function=stopBptt"},
		immediate=true)
public class DianneRNNCommands {

	private Dianne dianne;
	private DiannePlatform platform;
	private Learner learner;

	private TensorFactory factory;
	
	protected NeuralNetworkInstanceDTO nni;
	
	public void generate(String nnName, String start, int n){
		// forward of a rnn
		try {
			nni = platform.deployNeuralNetwork(nnName, "test rnn");
			NeuralNetwork nn = dianne.getNeuralNetwork(nni).getValue();
			
			String result = ""+start;
			
			for(int i=0;i<start.length()-1;i++){
				nextChar(nn, start.charAt(i));
			}
			
			char c = start.charAt(start.length()-1);
			for(int i=0;i<n;i++){
				c = nextChar(nn, c);
				result += c;
			}
			
			System.out.println(result);
			
		} catch(Exception e){
			e.printStackTrace();
		} finally {
			platform.undeployNeuralNetwork(nni);
		}
	}
	

	public void bptt(String nnName, String dataset, String... properties){
		try {
			Map<String, String> config = createLearnerConfig(properties);
			
			nni = platform.deployNeuralNetwork(nnName);
			learner.learn(dataset, config, nni);
		} catch(Exception e){
			e.printStackTrace();
			platform.undeployNeuralNetwork(nni);
		} 
	}

	public void stopBptt(){
		this.learner.stop();
		platform.undeployNeuralNetwork(nni);
	}
	
	private Map<String, String> createLearnerConfig(String[] properties){
		Map<String, String> config = new HashMap<String, String>();
		// defaults
		config.put("sequenceLength", "20");
		config.put("criterion", "MSE");
		config.put("learningRate", "0.01");
		config.put("momentum", "0.9");

		for(String property : properties){
			String[] p = property.split("=");
			if(p.length==2){
				config.put(p[0].trim(), p[1].trim());
			}
		}
		
		return config;
	}
	
	
	
	private char nextChar(NeuralNetwork nn, char current){
		// construct input tensor
		String[] labels = nn.getOutputLabels();
		if(labels==null){
			throw new RuntimeException("Neural network "+nn.getNeuralNetworkInstance().name+" is not trained and has no labels");
		}
		Tensor in = factory.createTensor(labels.length);
		in.fill(0.0f);
		int index = 0;
		for(int i=0;i<labels.length;i++){
			if(labels[i].charAt(0)==current){
				index = i;
				break;
			}
		}
		in.set(1.0f, index);
		
		// forward
		Tensor out = nn.forward(in);
		
		// select next, for now arg max, better sample here?
		int o = factory.getTensorMath().argmax(out);
		
		return labels[o].charAt(0);
	}
	
	@Reference
	void setTensorFactory(TensorFactory tf){
		this.factory = tf;
	}
	
	@Reference
	void setDianne(Dianne d){
		this.dianne = d;
	}
	
	@Reference
	void setDiannePlatform(DiannePlatform p){
		this.platform = p;
	}
	
	@Reference(target="(dianne.learner.type=recurrent)")
	void setLearner(Learner l){
		this.learner = l;
	}
}
