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
import be.iminds.iot.input.joystick.api.JoystickListener;
import be.iminds.iot.input.keyboard.api.KeyboardEvent;
import be.iminds.iot.input.keyboard.api.KeyboardListener;
import be.iminds.iot.robot.api.rover.Rover;

public class RoverOutput extends ThingOutput implements JoystickListener, KeyboardListener {
	
	private enum Mode {
		IGNORE,
		DISCRETE,
		CONTINUOUS,
		ANY
	}
	
	private Rover rover;
	
	private float speed = 0.15f;
	
	private float throttle = 0;
	private float yaw = 0;
	
	private ServiceRegistration registration;
	
	private Mode mode = Mode.ANY;
	
	public RoverOutput(UUID id, String name, BundleContext context){
		super(id, name, "Youbot");
		
		String s = context.getProperty("be.iminds.iot.dianne.rover.speed");
		if(s!=null){
			speed = Float.parseFloat(s);
		}
	}
	
	public void setRover(Rover r){
		this.rover = r;
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
		
		// TODO this code is replicated from the Environment to have same behavior
		// Should this somehow be merged together?
		int outputs = output.size(0);
		if(outputs >= 3 && (mode == Mode.DISCRETE || mode == Mode.ANY)){
			// treat as discrete outputs
			int action = TensorOps.argmax(output);
			
			switch(action){
			case 0:
				// forward
				throttle = speed;
				yaw = 0;
				break;
			case 1:
				// left
				throttle = speed;
				yaw = -1;
				break;
			case 2:
				// right
				throttle = speed;
				yaw = 1;
				break;
			case 3:
				// break
				throttle = 0;
				yaw = 0;
				break;
			case 4:
				// left break
				throttle = 0;
				yaw = -1;
				break;
			case 5:
				// right break
				throttle = 0;
				yaw = 1;
				break;
			}

		} else if(outputs == 2 && (mode == Mode.CONTINUOUS || mode == Mode.ANY)) {
			float[] action = output.get();
			// treat as continuous outputs
			throttle = action[0];
			yaw = action[1];
		} 

		rover.move(throttle, yaw);

	}

	@Override
	public void onError(UUID moduleId, ModuleException e, String... tags) {
	}

	public void connect(UUID nnId, UUID outputId, BundleContext context){
		if(!isConnected()){
			registration = context.registerService(new String[]{JoystickListener.class.getName(),KeyboardListener.class.getName()}, this, new Hashtable<>());
		}
		
		super.connect(nnId, outputId, context);
	}
	
	public void disconnect(UUID moduleId, UUID outputId){
		// stop rover on disconnect
		super.disconnect(moduleId, outputId);
		
		if(!isConnected()){
			rover.stop();
			
			registration.unregister();
		}
	}

	@Override
	public void onEvent(JoystickEvent e) {
		switch(e.type){
		case BUTTON_X_PRESSED:
			rover.stop();
			mode = Mode.IGNORE;
			System.out.println("Ignore any neural net robot control signals");
			break;
		case BUTTON_Y_PRESSED:
			mode = Mode.DISCRETE;
			System.out.println("Accept only discrete neural net robot control signals");
			break;
		case BUTTON_B_PRESSED:
			mode = Mode.CONTINUOUS;
			System.out.println("Accept only continous neural net robot control signals");
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
			rover.stop();
			mode = Mode.IGNORE;
			System.out.println("Ignore any neural net robot control signals");
			break;
		case "2":
			mode = Mode.DISCRETE;
			System.out.println("Accept only discrete neural net robot control signals");
			break;
		case "3":
			mode = Mode.CONTINUOUS;
			System.out.println("Accept only continous neural net robot control signals");
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
