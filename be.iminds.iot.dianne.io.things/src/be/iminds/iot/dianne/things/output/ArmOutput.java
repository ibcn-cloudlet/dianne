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
package be.iminds.iot.dianne.things.output;

import java.util.UUID;

import org.osgi.framework.BundleContext;

import be.iminds.iot.dianne.api.nn.module.ModuleException;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.robot.api.arm.Arm;

public class ArmOutput extends ThingOutput {
	
	private BundleContext context;
	
	private Arm arm;
	
	private final Tensor value = new Tensor(new float[]{0.0f}, 1);
	private final Tensor position = new Tensor(3);
	
	private volatile boolean init = false;
	private volatile boolean active = true;
	
	private float threshold = 0.25f;
	private float hoverHeight = 0.15f;
	private float gripHeight = 0.085f;
	
	public ArmOutput(UUID id, String name){
		super(id, name, "Arm");
	}
	
	public void setArm(Arm a){
		this.arm = a;
	}
	
	@Override
	public void onForward(final UUID moduleId, final Tensor output, final String... tags) {
		if(!init){
			this.arm.openGripper()
				.then(p -> arm.moveTo(0.4f, 0.0f, 0.4f))
				.then(p -> {init = true; active = false; return p;});
		}
		
		if(output.size()==1){
			output.copyInto(value);
		} else if(output.size()==3){
			output.copyInto(position);
		} else {
			// invalid output dimension
			return;
		}
		
		
		if(value.get(0) >= threshold){
			if(!active){
				active = true;

				arm.moveTo(position.get(0), position.get(1), hoverHeight)
					.then(p -> arm.moveTo(position.get(0), position.get(1), gripHeight))
					.then(p -> arm.closeGripper())
					.then(p -> arm.moveTo(0.4f, 0.0f, 0.4f))
					.then(p -> arm.openGripper())
					.then(p -> {active = false; return p;});
			}
		}
	}

	@Override
	public void onError(UUID moduleId, ModuleException e, String... tags) {
	}

	@Override
	public void connect(UUID nnId, UUID outputId, BundleContext context){
		if(this.context == null)
			activate(context);
		
		super.connect(nnId, outputId, context);
	}
	
	@Override
	public void disconnect(UUID moduleId, UUID outputId){
		// stop youbot on disconnect
		super.disconnect(moduleId, outputId);
		
		if(registrations.isEmpty()){
			arm.stop();
			arm.reset();
		}
	}
	
	private void activate(BundleContext context){
		String s = context.getProperty("be.iminds.iot.dianne.youbot.arm.hoverHeight");
		if(s!=null){
			hoverHeight = Float.parseFloat(s);
		}
		
		s = context.getProperty("be.iminds.iot.dianne.youbot.arm.gripHeight");
		if(s!=null){
			gripHeight = Float.parseFloat(s);
		}
		
		s = context.getProperty("be.iminds.iot.dianne.youbot.arm.valueThreshold");
		if(s!=null){
			threshold = Float.parseFloat(s);
		}
	}

}
