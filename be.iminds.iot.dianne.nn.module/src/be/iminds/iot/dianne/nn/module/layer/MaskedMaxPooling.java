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
package be.iminds.iot.dianne.nn.module.layer;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class MaskedMaxPooling extends AbstractModule {
	
	private List<Tensor> masks;
	
	public MaskedMaxPooling(TensorFactory factory,
			int noInputs, String maskIndices){
		super(factory);
		initMasks(noInputs, maskIndices);
	}
	
	public MaskedMaxPooling(TensorFactory factory, UUID id, 
			int noInputs, String maskIndices){
		super(factory, id);
		initMasks(noInputs, maskIndices);
	}
	
	private void initMasks(int noInputs, String maskIndices){
		String[] m = maskIndices.split(";");
		int noOutputs = m.length+1;
		masks = new ArrayList<>(noOutputs);
		
		Tensor other = factory.createTensor(noInputs);
		other.fill(1.0f);
		for(int i=0;i<m.length;i++){
			Tensor mask = factory.createTensor(noInputs);
			mask.fill(0.0f);
			
			String[] indices = m[i].split(",");
			for(String index : indices){
				int k = Integer.parseInt(index);
				mask.set(1.0f, k);
				other.set(0.0f, k);
			}
			
			masks.add(mask);
		}
		masks.add(other);
	}

	@Override
	protected void forward() {
		if(output==null || !output.hasDim(masks.size())){
			output = factory.createTensor(masks.size());
		}
		
		Tensor temp = null;
		for(int i=0;i<masks.size();i++){
			temp = factory.getTensorMath().cmul(temp, input, masks.get(i));
			float max = factory.getTensorMath().max(temp);
			output.set(max, i);
		}
	}

	@Override
	protected void backward() {
		throw new UnsupportedOperationException();
	}
	
}
