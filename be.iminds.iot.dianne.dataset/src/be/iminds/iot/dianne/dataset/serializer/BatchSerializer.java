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
package be.iminds.iot.dianne.dataset.serializer;

import org.osgi.service.component.annotations.Component;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import be.iminds.iot.dianne.api.dataset.Batch;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * Special serializer required to make sure all mappings in Batch objects are correct
 * @author tverbele
 *
 */
@Component(service = Serializer.class, property = { 
		"aiolos.export=false",
		"kryo.serializer.class=be.iminds.iot.dianne.api.dataset.Batch", 
		"kryo.serializer.id=1000" })
public class BatchSerializer extends Serializer<Batch> {

	@Override
	public Batch read(Kryo kryo, Input input, Class<Batch> batch) {
		Tensor in = kryo.readObject(input, Tensor.class);
		Tensor target = kryo.readObject(input, Tensor.class);
		return new Batch(in, target);
	}

	@Override
	public void write(Kryo kryo, Output output, Batch batch) {
		try {
			kryo.writeObject(output, batch.input);
			kryo.writeObject(output, batch.target);
		} catch(Throwable t){
			t.printStackTrace();
			throw t;
		}
	}

}
