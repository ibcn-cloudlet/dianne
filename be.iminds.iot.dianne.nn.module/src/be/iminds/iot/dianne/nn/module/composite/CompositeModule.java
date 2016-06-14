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
package be.iminds.iot.dianne.nn.module.composite;

import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.AbstractTrainableModule;
import be.iminds.iot.dianne.api.nn.module.Composite;
import be.iminds.iot.dianne.api.nn.module.Memory;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.tensor.Tensor;

public class CompositeModule extends AbstractTrainableModule implements Composite, Memory {

	private final NeuralNetwork nn;
	private final LinkedHashMap<UUID, Integer> parameterMapping;

	private final Tensor memory;
	
	private Map<UUID, Tensor> deltas;
	
	private Tensor in;
	
	public CompositeModule(UUID id, Tensor parameters, Tensor memory, NeuralNetwork nn, LinkedHashMap<UUID, Integer> parameterMapping){
		super(id, parameters);
		this.nn = nn;
		this.memory = memory;
		this.parameterMapping = parameterMapping;
	}
	
	public void initDeltaParameters(Tensor t){
		if(t==null){
			deltaParameters = new Tensor(parameters.dims());
		} else {
			// TODO check size?
			deltaParameters = t;
		}
		
		deltas = new HashMap<>();
		int offset = 0;
		Iterator<Entry<UUID, Integer>> it = parameterMapping.entrySet().iterator();
		while(it.hasNext()){
			Entry<UUID, Integer> e = it.next();
			int size = e.getValue();
			Tensor narrowed = deltaParameters.narrow(0, offset, size);
			nn.getTrainables().get(e.getKey()).initDeltaParameters(narrowed);
			deltas.put(e.getKey(), narrowed);
			offset += size;
		}
	}
	
	
	@Override 
	public void randomize(){
		nn.getTrainables().values().stream().forEach(m -> m.randomize());
	}
	
	@Override
	protected void forward() {
		// for now only supported for nns with one input and one output
		// TODO for now we need to take a copy of the input, for avoiding synchronisation issues
		// should be cleaner if we could pass it to next as if next input was next module?
		in = input.copyInto(in);
		
		output = nn.forward(in);
	}

	@Override
	protected void backward() {
		if(deltaParameters==null){
			initDeltaParameters(null);
		}
		gradInput = nn.backward(gradOutput);
	}

	@Override
	public void accGradParameters() {
		nn.getTrainables().entrySet().forEach(e -> e.getValue().accGradParameters());
		
		// copy to composite deltaParameters (in case both same Tensor no actual copy will be done - to let it work for remotely deployed modules) 
		nn.getTrainables().entrySet().forEach(e -> e.getValue().setDeltaParameters(deltas.get(e.getKey())));

	}
	
	@Override
	public void zeroDeltaParameters() {
		if(deltaParameters!=null){
			deltaParameters.fill(0.0f);
		}
		
		// also forward to composing modules
		nn.getTrainables().values().forEach(m -> m.zeroDeltaParameters());
	}

	@Override
	public void triggerForward(final String... tags) {
		nn.getMemories().values().forEach(m -> m.triggerForward(tags));
	}

	@Override
	public void triggerBackward(final String... tags) {
		nn.getMemories().values().forEach(m -> m.triggerBackward(tags));
	}

	@Override
	public Tensor getMemory() {
		return memory;
	}

	@Override
	public void setMemory(Tensor memory) {
		memory.copyInto(this.memory);
	}

	@Override
	public NeuralNetworkInstanceDTO getNNi() {
		return nn.getNeuralNetworkInstance();
	}

}
