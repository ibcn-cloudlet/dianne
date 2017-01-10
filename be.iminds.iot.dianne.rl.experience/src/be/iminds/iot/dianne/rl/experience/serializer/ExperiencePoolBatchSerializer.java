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
package be.iminds.iot.dianne.rl.experience.serializer;

import org.osgi.service.component.annotations.Component;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolBatch;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * Special serializer required to make sure all mappings in ExperiencePoolBatch objects are correct
 * @author tverbele
 *
 */
@Component(service = Serializer.class, property = { 
		"aiolos.export=false",
		"kryo.serializer.class=be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolBatch", 
		"kryo.serializer.id=1001" })
public class ExperiencePoolBatchSerializer extends Serializer<ExperiencePoolBatch> {

	@Override
	public ExperiencePoolBatch read(Kryo kryo, Input input, Class<ExperiencePoolBatch> batch) {
		Tensor state = kryo.readObject(input, Tensor.class);
		Tensor action = kryo.readObject(input, Tensor.class);
		Tensor reward = kryo.readObject(input, Tensor.class);
		Tensor nextState = kryo.readObject(input, Tensor.class);
		Tensor terminal = kryo.readObject(input, Tensor.class);
		return new ExperiencePoolBatch(state, action, reward, nextState, terminal);
	}

	@Override
	public void write(Kryo kryo, Output output, ExperiencePoolBatch batch) {
		try {
			kryo.writeObject(output, batch.input);
			kryo.writeObject(output, batch.target);
			kryo.writeObject(output, batch.reward);
			kryo.writeObject(output, batch.nextState);
			kryo.writeObject(output, batch.terminal);
		} catch(Throwable t){
			t.printStackTrace();
			throw t;
		}
	}

}
