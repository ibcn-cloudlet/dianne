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
package be.iminds.iot.dianne.tensor.impl.java;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

@Component(service=Serializer.class, property={"aiolos.export=false",
	"kryo.serializer.class=be.iminds.iot.dianne.tensor.impl.java.JavaTensor",
	"kryo.serializer.id=100"})
public class JavaTensorSerializer extends Serializer<Tensor> {

	private TensorFactory factory;
	
	@Reference
	void setTensorFactory(TensorFactory factory){
		this.factory = factory;
	}
	
	@Override
	public Tensor read(Kryo kryo, Input input, Class<Tensor> tensor) {
		int noDims = input.readInt();
		int[] dims = input.readInts(noDims);
		int length = input.readInt();
		float[] data = input.readFloats(length);
		return factory.createTensor(data, dims);
	}

	@Override
	public void write(Kryo kryo, Output output, Tensor tensor) {
		output.writeInt(tensor.dims().length);
		output.writeInts(tensor.dims());
		output.writeInt(tensor.size());
		output.writeFloats(tensor.get());
	}

}
