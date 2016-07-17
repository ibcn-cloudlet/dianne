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
package be.iminds.iot.dianne.tensor;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;

@Component(immediate=true)
public class NativeTensorLoader {

	static {
		try {
			System.loadLibrary("Tensor");
		} catch (final UnsatisfiedLinkError e) {
		    System.err.println("Native code library Tensor failed to load. \n"+ e);
		    throw e;
		}
	}
	
	@Activate
	public void activate(BundleContext context){
		int device = -1;
		if(context != null){
			String d = context.getProperty("be.iminds.iot.dianne.tensor.device");
			if(d != null){
				device = Integer.parseInt(d);
			}
		}
		
		init(device);
	}
	
	@Deactivate()
	public void deactivate(){
		cleanup();
	}
	
	// set GPU device id in case of multiple GPUs on machine!
	private native void init(int device);
	
	private native void cleanup();
	
}
