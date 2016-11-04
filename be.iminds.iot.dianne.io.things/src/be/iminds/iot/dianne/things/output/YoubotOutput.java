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
import be.iminds.iot.dianne.tensor.TensorOps;
import be.iminds.iot.robot.api.Arm;
import be.iminds.iot.robot.api.OmniDirectional;

public class YoubotOutput extends ThingOutput {
	
	private OmniDirectional base;
	private Arm arm;
	
	private float speed = 0.1f;
	private float threshold = 0.01f;
	
	private float vx = 0;
	private float vy = 0;
	private float va = 0;
	private boolean grip = false;
	
	private volatile boolean skip = false;
	private volatile boolean stop = false;
	
	public YoubotOutput(UUID id, String name){
		super(id, name, "Youbot");
	}
	
	public void setBase(OmniDirectional b){
		this.base = b;
	}
	
	public void setArm(Arm a){
		this.arm = a;
	}
	
	@Override
	public void onForward(final UUID moduleId, final Tensor output, final String... tags) {
		if(output.dim() != 1){
			System.out.println("Wrong output dimensions");
			return;
		}
		
		if(skip || stop){
			return;
		}
		
		// TODO this code is replicated from the Environment to have same behavior
		// Should this somehow be merged together?
		int outputs = output.size(0);
		if(outputs == 7){
			// treat as discrete outputs
			int action = TensorOps.argmax(output);
			grip = false;
			switch(action){
			case 0:
				vy = speed;
				break;
			case 1:
				vy = -speed;
				break;
			case 2:
				vx = speed;
				break;
			case 3:
				vx = -speed;
				break;
			case 4:
				va = 2*speed;
				break;
			case 5:
				va = -2*speed;
				break;	
			case 6:
				grip = true;
			}
			
		} else if(outputs == 3) {
			float[] action = output.get();
			// treat as continuous outputs
			if(  action[0] < threshold
				&& action[1] < threshold
				&& action[2] < threshold){
				// grip	
				grip = true;
			} else {
				// move
				grip = false;
				vx = action[0]*speed;
				vy = action[1]*speed;
				va = action[2]*speed*2;
			}
		}
		
		
		if(grip){
			base.stop();	
			arm.openGripper()
					.then(p -> arm.setPositions(2.92f, 0.0f, 0.0f, 0.0f, 2.875f))
					.then(p -> arm.setPositions(2.92f, 1.76f, -1.37f, 2.55f))
					.then(p -> arm.closeGripper())
					.then(p -> arm.setPositions(0.01f, 0.8f))
					.then(p -> arm.setPositions(0.01f, 0.8f, -1f, 2.9f))
					.then(p -> arm.openGripper())
					.then(p -> arm.setPosition(1, -1.3f))
					.then(p -> arm.reset()).then(p -> {skip = false; return null;});
			skip = true;
		} else {
			base.move(vx, vy, va);
		}
	}

	@Override
	public void onError(UUID moduleId, ModuleException e, String... tags) {
	}

	public void connect(UUID nnId, UUID outputId, BundleContext context){
		stop = false;
		super.connect(nnId, outputId, context);
	}
	
	public void disconnect(){
		// stop youbot on disconnect
		stop = true;
		super.disconnect();
		
		base.stop();
		arm.stop();
	}
}
