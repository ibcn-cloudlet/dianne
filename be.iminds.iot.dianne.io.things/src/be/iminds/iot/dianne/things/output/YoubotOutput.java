/*******************************************************************************
act null FetchCan null strategy=RandomActionStrategy discrete=true trace=true maxActions=100
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

import java.util.Hashtable;
import java.util.UUID;

import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceRegistration;

import be.iminds.iot.dianne.api.nn.module.ModuleException;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;
import be.iminds.iot.input.joystick.api.JoystickEvent;
import be.iminds.iot.input.joystick.api.JoystickEvent.JoystickButton;
import be.iminds.iot.input.joystick.api.JoystickListener;
import be.iminds.iot.input.keyboard.api.KeyboardEvent;
import be.iminds.iot.input.keyboard.api.KeyboardListener;
import be.iminds.iot.robot.api.arm.Arm;
import be.iminds.iot.robot.api.omni.OmniDirectional;

public class YoubotOutput extends ThingOutput implements JoystickListener, KeyboardListener {
	
	private enum Mode {
		IGNORE,
		DISCRETE,
		DISCRETE_SOFTMAX,
		CONTINUOUS,
		STOCHASTIC,
		ANY
	}
	
	private OmniDirectional base;
	private Arm arm;
	
	private float speed = 0.1f;
	private float gripThreshold = 0.05f;
	private float ignoreGripThreshold = -0.02f;
	
	private float vx = 0;
	private float vy = 0;
	private float va = 0;
	private boolean grip = false;

	private Tensor sample = new Tensor(3);
	
	private volatile boolean skip = false;
	private volatile boolean stop = false;
	
	private ServiceRegistration registration;
	
	private Mode mode = Mode.ANY;
	
	public YoubotOutput(UUID id, String name, BundleContext context){
		super(id, name, "Youbot");
		
		String s = context.getProperty("be.iminds.iot.dianne.youbot.speed");
		if(s!=null){
			speed = Float.parseFloat(s);
		}
		
		s = context.getProperty("be.iminds.iot.dianne.youbot.gripThreshold");
		if(s!=null){
			gripThreshold = Float.parseFloat(s);
		}
		
		s = context.getProperty("be.iminds.iot.dianne.youbot.ignoreGripThreshold");
		if(s!=null){
			ignoreGripThreshold = Float.parseFloat(s);
		}

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
		
		if(mode == Mode.IGNORE){
			return;
		}
		
		if(skip || stop){
			return;
		}
		
		// TODO this code is replicated from the Environment to have same behavior
		// Should this somehow be merged together?
		int outputs = output.size(0);
		if(outputs == 7 && (mode == Mode.DISCRETE || mode == Mode.DISCRETE_SOFTMAX || mode == Mode.ANY)){
			// treat as discrete outputs
			int action = TensorOps.argmax(output);
			
			float sum = TensorOps.sum(TensorOps.exp(null, output));
			Tensor narrowed = output.narrow(0, 5);
			if(Math.abs(1.0f - sum) < 0.001){
				if(mode == Mode.DISCRETE)
					return;
				
				// coming from logsoftmax policy (should we sample?)
				if(action == 6 && output.get(6) < ignoreGripThreshold){
					action = TensorOps.argmax(narrowed);
				}
			} else {
				// DQN network, values are Q values
				if(mode == Mode.DISCRETE_SOFTMAX){
					return;
				}
				
				// if gripThreshold specified, use that one instead of grip Q value (which is hard to train)
				if(gripThreshold > 0 && TensorOps.dot(narrowed, narrowed) < gripThreshold){
					action = 6;
				}
			}
			
			grip = false;
			switch(action){
			case 0:
				vx = 0;
				vy = speed;
				va = 0;
				break;
			case 1:
				vx = 0;
				vy = -speed;
				va = 0;
				break;
			case 2:
				vx = speed;
				vy = 0;
				va = 0;
				break;
			case 3:
				vx = -speed;
				vy = 0;
				va = 0;
				break;
			case 4:
				vx = 0;
				vy = 0;
				va = 2*speed;
				break;
			case 5:
				vx = 0;
				vy = 0;
				va = -2*speed;
				break;	
			case 6:
				grip = true;
			}
			
		} else if(outputs == 3 && (mode == Mode.CONTINUOUS || mode == Mode.ANY)) {
			float[] action = output.get();
			// treat as continuous outputs
			if(TensorOps.dot(output, output) < gripThreshold){
				// grip	
				grip = true;
			} else {
				// move
				grip = false;
				vx = action[0]*speed;
				vy = action[1]*speed;
				va = action[2]*speed*2;
			}
		} else if(outputs == 6 && (mode == Mode.STOCHASTIC || mode == Mode.ANY)) {
			sample.randn();
			
			TensorOps.cmul(sample, sample, output.narrow(0, 3, 3));
			TensorOps.add(sample, sample, output.narrow(0, 0, 3));
		
			float[] action = sample.get();
			// treat as continuous outputs
			if(TensorOps.dot(sample, sample) < gripThreshold){
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
		if(!isConnected()){
			stop = false;
			registration = context.registerService(new String[]{JoystickListener.class.getName(),KeyboardListener.class.getName()}, this, new Hashtable<>());
		}
		
		super.connect(nnId, outputId, context);
	}
	
	public void disconnect(UUID moduleId, UUID outputId){
		// stop youbot on disconnect
		super.disconnect(moduleId, outputId);
		
		if(!isConnected()){
			stop = true;
	
			base.stop();
			arm.stop();
			
			registration.unregister();
		}
	}

	@Override
	public void onEvent(JoystickEvent e) {
		switch(e.type){
		case BUTTON_X_PRESSED:
			base.stop();
			mode = Mode.IGNORE;
			System.out.println("Ignore any neural net robot control signals");
			break;
		case BUTTON_Y_PRESSED:
			mode = Mode.DISCRETE;
			System.out.println("Accept only discrete neural net robot control signals");
			break;
		case BUTTON_A_PRESSED:
			mode = Mode.DISCRETE_SOFTMAX;
			System.out.println("Accept only discrete softmax neural net robot control signals");
			break;	
		case BUTTON_B_PRESSED:
			if(e.isPressed(JoystickButton.BUTTON_R1)){
				mode = Mode.STOCHASTIC;
				System.out.println("Accept only stochastic continuous neural net robot control signals");
			} else {
				mode = Mode.CONTINUOUS;
				System.out.println("Accept only continous neural net robot control signals");
			}
			break;
		default:
		}
	}

	@Override
	public void onEvent(KeyboardEvent e) {
		if(e.type!=KeyboardEvent.Type.PRESSED)
			return;
		
		switch(e.key){
		case "1":
			base.stop();
			mode = Mode.IGNORE;
			System.out.println("Ignore any neural net robot control signals");
			break;
		case "2":
			mode = Mode.DISCRETE;
			System.out.println("Accept only discrete neural net robot control signals");
			break;
		case "3":
			mode = Mode.DISCRETE_SOFTMAX;
			System.out.println("Accept only discrete softmax neural net robot control signals");
			break;	
		case "4":
			mode = Mode.CONTINUOUS;
			System.out.println("Accept only continous neural net robot control signals");
			break;
		case "5":
			mode = Mode.STOCHASTIC;
			System.out.println("Accept only stochastic continuous neural net robot control signals");
			break;	
		case "0":
			mode = Mode.ANY;
			System.out.println("Accept any robot control signals");
			break;
		default:
			break;
		}
	}
}
