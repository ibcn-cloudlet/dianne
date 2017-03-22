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
package be.iminds.iot.dianne.dataset;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;

import be.iminds.iot.dianne.api.dataset.Batch;
import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.RawBatch;
import be.iminds.iot.dianne.api.dataset.RawBatchedSequence;
import be.iminds.iot.dianne.api.dataset.RawSample;
import be.iminds.iot.dianne.api.dataset.RawSequence;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.api.dataset.Sequence;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolBatch;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.api.rl.dataset.RawBatchedExperiencePoolSequence;
import be.iminds.iot.dianne.api.rl.dataset.RawExperiencePoolBatch;
import be.iminds.iot.dianne.api.rl.dataset.RawExperiencePoolSample;
import be.iminds.iot.dianne.api.rl.dataset.RawExperiencePoolSequence;

/**
 * Proxy class to optimize the behavior of querying datasets that are located on a remote machine
 * 
 * @author tverbele
 *
 */
public class RemoteDatasetProxy implements InvocationHandler {

	private final Dataset proxied;
	
	public RemoteDatasetProxy(Dataset d){
		this.proxied = d;
	}
	
	@Override
	public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
		if(method.getParameterCount() > 0 
				&& method.getReturnType().equals(method.getParameterTypes()[0])
				&& args[0] != null){
			// get the raw data and copy it into the provided result object
			// on the caller side
			Class[] parameterTypes = new Class[args.length-1];
			Object[] arguments = new Object[args.length-1];
			for(int i=0;i<parameterTypes.length;i++){
				parameterTypes[i] = method.getParameterTypes()[i+1];
				arguments[i] = args[i+1];
			}
			String rawMethodName = "getRaw"+method.getName().substring(3);
			
			Method rawMethod = proxied.getClass().getMethod(rawMethodName, parameterTypes);
			Object rawResult = rawMethod.invoke(proxied, arguments);
			if(rawResult instanceof RawExperiencePoolSample){
				return ((RawExperiencePoolSample)rawResult).copyInto((ExperiencePoolSample)args[0]);
			} else if(rawResult instanceof RawExperiencePoolBatch){
				return ((RawExperiencePoolBatch)rawResult).copyInto((ExperiencePoolBatch)args[0]);
			} else if(rawResult instanceof RawSample){
				return ((RawSample)rawResult).copyInto((Sample)args[0]);
			} else if(rawResult instanceof RawBatch){
				return ((RawBatch)rawResult).copyInto((Batch)args[0]);
			} else if(rawResult instanceof RawExperiencePoolSequence){
				return ((RawExperiencePoolSequence)rawResult).copyInto((Sequence)args[0]);
			} else if(rawResult instanceof RawBatchedExperiencePoolSequence){
				return ((RawBatchedExperiencePoolSequence)rawResult).copyInto((Sequence)args[0]);
			} else if(rawResult instanceof RawSequence){
				return ((RawSequence)rawResult).copyInto((Sequence)args[0]);
			} else if(rawResult instanceof RawBatchedSequence){
				return ((RawBatchedSequence)rawResult).copyInto((Sequence)args[0]);
			} else {
				throw new RuntimeException("Unsupported raw result "+rawResult);
			}
		}
		return method.invoke(proxied, args);
	}

}
