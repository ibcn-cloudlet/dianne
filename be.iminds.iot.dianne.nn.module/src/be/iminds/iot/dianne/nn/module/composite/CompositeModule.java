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
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class CompositeModule extends AbstractTrainableModule {

	private final NeuralNetwork nn;
	private final LinkedHashMap<UUID, Integer> parameterMapping;
	
	private Map<UUID, Tensor> deltas;
	
	public CompositeModule(TensorFactory factory, UUID id, Tensor parameters, NeuralNetwork nn, LinkedHashMap<UUID, Integer> parameterMapping){
		super(factory, id, parameters);
		this.nn = nn;
		this.parameterMapping = parameterMapping;
	}
	
	private void initDeltaParameters(){
		deltaParameters = factory.createTensor(parameters.dims());
		
		deltas = new HashMap<>();
		int offset = 0;
		Iterator<Entry<UUID, Integer>> it = parameterMapping.entrySet().iterator();
		while(it.hasNext()){
			Entry<UUID, Integer> e = it.next();
			int size = e.getValue();
			Tensor narrowed = deltaParameters.narrow(0, offset, size);
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
		input = input.copyInto(null);
		
		output = nn.forward(input);
	}

	@Override
	protected void backward() {
		if(deltaParameters==null){
			initDeltaParameters();
		}
		gradInput = nn.backward(gradOutput);
	}

	@Override
	public void accGradParameters() {
		// since delta parameters are kept in separate tensor here, this introduces some overhead
		// i.e. each time the delta of individual composing modules is copied over to this deltaparams
		// TODO fix this by also injecting our deltaparams tensor into the modules?
		nn.getTrainables().entrySet().forEach(e -> e.getValue().accGradParameters());
		
		// copy to composite deltaParameters 
		nn.getTrainables().entrySet().forEach(e -> e.getValue().getDeltaParameters().copyInto(deltas.get(e.getKey())));

	}
	
	@Override
	public void zeroDeltaParameters() {
		if(deltaParameters!=null){
			deltaParameters.fill(0.0f);
		}
		
		// also forward to composing modules
		nn.getTrainables().values().forEach(m -> m.zeroDeltaParameters());
	}

}
