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
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.UUID;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
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
				  "osgi.command.function=generate2",
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
				System.out.println("Failed to load nn parameters");
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
		} catch(Exception e){
			e.printStackTrace();
		}
	}
	
	public void bptt(String nnName){
		// training of a rnn by unfolding
		try {
			int length = 25;
			
			Criterion criterion = new MSECriterion(factory);
			
			Random rand = new Random();
			String sequence = "aaabbbcccdddeeeaaabbbcccdddeeeaaabbbcccdddeeeaaabbbcccdddeeeaaabbbcccdddeeeaaabbbcccdddeeeaaabbbcccdddeee";
			
			// find recurrent links to unfold
			List<RecurrentLink> recurrentLinks = findRecurrentLinks(platform.getAvailableNeuralNetwork(nnName));
		
			List<NeuralNetworkInstanceDTO> nnis = new ArrayList<>(length);
			List<NeuralNetwork> nns = new ArrayList<>(length);
			
			// create nn instances
			for(int i=0;i<length;i++){
				NeuralNetworkInstanceDTO nni = platform.deployNeuralNetwork(nnName);
				nnis.add(nni);
				NeuralNetwork nn = dianne.getNeuralNetwork(nni).getValue();
				nns.add(nn);
			}
			
			// unfold the next/prev of the recurrent links
			for(int i=0;i<length;i++){
				for(RecurrentLink l : recurrentLinks){
					// next should be forwarded to next nn in unfolded list
					Module m1 = nns.get(i).getModules().get(l.from.id);
					Module[] next = new Module[l.from.next.length];
					for(int j=0;j<next.length;j++){
						if(l.from.next[j].equals(l.to.id)){
							// forward to next nn
							if(i==length-1){
								next[j] = null;
							} else {
								next[j] = nns.get(i+1).getModules().get(l.from.next[j]);
							}
						} else {
							next[j] = nns.get(i).getModules().get(l.from.next[j]);
						}
					}
					m1.setNext(next);
					
					// previous should be backwarded to prev nn in unfolded list
					Module m2 = nns.get(i).getModules().get(l.to.id);
					Module[] prev = new Module[l.to.prev.length];
					for(int j=0;j<prev.length;j++){
						if(l.to.prev[j].equals(l.from.id)){
							// backward to prev nn
							if(i==0){
								prev[j] = null;
							} else {
								prev[j] = nns.get(i-1).getModules().get(l.to.prev[j]);
							}
						} else {
							prev[j] = nns.get(i).getModules().get(l.to.prev[j]);
						}
					}
					m2.setPrevious(prev);
				}
			}
			
			
			// initialize parameters
			nns.get(0).randomizeParameters();
			nns.get(0).storeParameters("test");
			
			
			int iterations = 1000;
			float learningRate = 0.01f;
			for(int l=0;l<iterations;l++){
	
				// load parameters
				Map<UUID, Tensor> previousParameters = null;
				for(int k=0;k<nns.size();k++){
					try {
						previousParameters = nns.get(k).loadParameters("test");
					}catch(Exception e){
						e.printStackTrace();
					}
				}
				
				// zero delta params
				for(int k=0;k<nns.size();k++){
					nns.get(k).getTrainables().values().stream().forEach(Trainable::zeroDeltaParameters);
				}
				
				// forward
				int offset = rand.nextInt(20);
				Tensor[] outputs = new Tensor[length];
				for(int i=0;i<length;i++){
					// forward the sequence
					Tensor out = nns.get(i).forward(asTensor(sequence.charAt(offset+i)));
					outputs[i] = out;
				}
				
				// backward
				float error = 0;
				for(int i=length-1;i>=0;i--){
					Tensor target = asTensor(sequence.charAt(offset+i+1));
					float e = criterion.error(outputs[i], target).get(0);
					error+=e;
					Tensor grad = criterion.grad(outputs[i], target);
					nns.get(i).backward(grad);
					
					// acc grad
					nns.get(i).getTrainables().values().stream().forEach(m -> m.accGradParameters());
					
				}
				System.out.println(error);

				// apply learning rate
				for(int k=0;k<nns.size();k++){
					nns.get(k).getTrainables().values().stream().forEach(
							m -> factory.getTensorMath().mul(m.getDeltaParameters(), m.getDeltaParameters(), -learningRate));
				}
				
				// update parameters
				for(int k=0;k<nns.size();k++){
					nns.get(k).getTrainables().values().stream().forEach(Trainable::updateParameters);
				}
				
				// store deltas
				for(int k=0;k<nns.size();k++){
					nns.get(k).storeDeltaParameters(previousParameters, "test");
				}
			
			}
			
			for(NeuralNetworkInstanceDTO nni : nnis){
				platform.undeployNeuralNetwork(nni);
			}
			
		} catch(Exception e){
			e.printStackTrace();
		}
		
	}
	
	private class RecurrentLink {
		ModuleDTO from;
		ModuleDTO to;
	}
	
	private List<RecurrentLink> findRecurrentLinks(NeuralNetworkDTO nn){
		List<RecurrentLink> links = new ArrayList<>();
		
		// should have only one input?
		ModuleDTO input = nn.modules.values().stream().filter(m -> m.type.equals("Input")).findFirst().get();
		visit(new ArrayList<UUID>(), links, input, nn.modules);
		
		return links;
	}
	
	private void visit(List<UUID> visited, List<RecurrentLink> links, ModuleDTO toVisit, Map<UUID, ModuleDTO> modules){
		visited.add(toVisit.id);
		if(toVisit.next==null){
			return;
		}
		for(UUID nxt : toVisit.next){
			if(visited.contains(nxt)){
				RecurrentLink l = new RecurrentLink();
				l.from = toVisit;
				l.to = modules.get(nxt);
				links.add(l);
			} else {
				visit(visited, links, modules.get(nxt), modules);
			}
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
