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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.UUID;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.nn.learn.criterion.MSECriterion;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

/**
 * Separate component for learn commands ... should be moved to the command bundle later on
 */
@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=generate",
				  "osgi.command.function=bptt"},
		immediate=true)
public class DianneRNNCommands {

	private Dianne dianne;
	private DiannePlatform platform;
	private TensorFactory factory;
	
	private final String chars = "abcde";
	
	public void generate(String nnName, char start, int n){
		// forward of a rnn
		try {
			NeuralNetworkInstanceDTO nni = platform.deployNeuralNetwork(nnName, "test rnn");
			NeuralNetwork nn = dianne.getNeuralNetwork(nni).getValue();
			try {
				nn.loadParameters("test");
			} catch(Exception e){
				System.out.println("Failed loading parameters.");
			}
			
			String result = ""+start;
			char c = start;
			for(int i=0;i<n;i++){
				Tensor in = asTensor((char)c);
				Tensor out = nn.forward(in);
				c = asChar(out);
				result += c;
			}
			
			System.out.println(result);
			
			platform.undeployNeuralNetwork(nni);
		} catch(Exception e){
			e.printStackTrace();
		}
	}
	
	public void bptt(String nnName){
		bptt(nnName, 25, 1000);
	}
	
	public void bptt(String nnName, int length, int iterations){
		// training of a rnn by unfolding
		try {
			Criterion criterion = new MSECriterion(factory);
			//Criterion criterion = new NLLCriterion(factory);
			
			Random rand = new Random();
			String sequence = "aaabbbcccdddeeeaaabbbcccdddeeeaaabbbcccdddeeeaaabbbcccdddeeeaaabbbcccdddeeeaaabbbcccdddeeeaaabbbcccdddeee";
			
			NeuralNetworkInstanceDTO nni = platform.deployNeuralNetwork(nnName);
			NeuralNetwork nn = dianne.getNeuralNetwork(nni).getValue();
			
			
			// initialize parameters
			nn.randomizeParameters();
			nn.storeParameters("test");
			
			
			float learningRate = 0.001f;
			for(int l=0;l<iterations;l++){
	
				// load parameters
				Map<UUID, Tensor> previousParameters = nn.loadParameters("test");
				
				// zero delta params
				nn.getTrainables().values().stream().forEach(Trainable::zeroDeltaParameters);
				
				// keep all memories intermediate states
				Map<UUID, Tensor[]> memories = new HashMap<UUID, Tensor[]>();
				nn.getMemories().entrySet().forEach(e -> {
					Tensor[] mems = new Tensor[length];
					for(int i=0;i<length;i++){
						mems[i] = e.getValue().getMemory().copyInto(null);
					}
					memories.put(e.getKey(), mems);
				});
				
				// forward the sequence
				int offset = rand.nextInt(20);
				Tensor[] inputs = new Tensor[length];
				Tensor[] outputs = new Tensor[length];
				for(int i=0;i<length;i++){
					// store input and memory
					inputs[i] = asTensor(sequence.charAt(offset+i));
					final int index = i;
					nn.getMemories().entrySet().forEach(e ->{
						e.getValue().getMemory().copyInto(memories.get(e.getKey())[index]);
					});
					
					// forward
					Tensor out = nn.forward(inputs[i]);
					
					// store output
					outputs[i] = out;

				}

				
				// backward
				float error = 0;
				for(int i=length-1;i>=0;i--){
					Tensor target = asTensor(sequence.charAt(offset+i+1));
					float err = criterion.error(outputs[i], target).get(0);
					error+=err;
					Tensor grad = criterion.grad(outputs[i], target);
					
					// first forward again with correct state and memories
					final int index = i;
					nn.getMemories().entrySet().forEach(e -> {
						e.getValue().setMemory(memories.get(e.getKey())[index]);
					});
					nn.forward(inputs[i]);
					
					// TODO set grad to zero for all intermediates?
					//if(i!=length-1){
					//	grad.fill(0.0f);
					//}
					nn.backward(grad);
					
					// acc grad
					nn.getTrainables().values().stream().forEach(m -> m.accGradParameters());
					
				}
				System.out.println(error);

				// apply learning rate
				nn.getTrainables().values().stream().forEach(
						m -> factory.getTensorMath().mul(m.getDeltaParameters(), m.getDeltaParameters(), -learningRate));
				
				// update parameters
				nn.getTrainables().values().stream().forEach(Trainable::updateParameters);
				
				// store deltas
				nn.storeDeltaParameters(previousParameters, "test");
				
			
			}
			
			platform.undeployNeuralNetwork(nni);
			
		} catch(Exception e){
			e.printStackTrace();
		}
		
	}
	
	private Tensor asTensor(char c){
		int index = chars.indexOf(c);
		Tensor t = factory.createTensor(chars.length());
		t.fill(0.0f);
		t.set(1.0f, index);
		return t;
	}
	
	private char asChar(Tensor t){
		int index = factory.getTensorMath().argmax(t);
		return chars.charAt(index);
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
}
