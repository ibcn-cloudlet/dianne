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

import be.iminds.iot.dianne.api.dataset.Dataset;

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
		if(method.getParameterCount() > 0 && method.getReturnType().equals(method.getParameterTypes()[0])){
			// this is a method that has as first arg the return type
			//... in case of a remote call, set parameter[0] to null
			args[0] = null;
		}
		return method.invoke(proxied, args);
	}

}
