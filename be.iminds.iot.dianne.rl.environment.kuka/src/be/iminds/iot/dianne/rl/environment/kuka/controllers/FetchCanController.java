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
package be.iminds.iot.dianne.rl.environment.kuka.controllers;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.rl.agent.ActionController;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.input.keyboard.api.KeyboardEvent;
import be.iminds.iot.input.keyboard.api.KeyboardListener;

/**
 * Listens to keyboard events and uses these to steer FetchCan ActionController
 *
 */
@Component()
public class FetchCanController implements KeyboardListener {

	private ActionController controller;
	
	@Reference(target="(environment=FetchCan)")
	void setActionController(ActionController a) {
		this.controller = a;
	}
	
	@Override
	public void onEvent(KeyboardEvent e) {
		if(e.type != KeyboardEvent.Type.PRESSED)
			return;
		
		Tensor action = new Tensor(7);
		action.fill(0.0f);
		
		switch(e.key){
		case "a":
			action.set(1.0f, 0);
			break;
		case "d":
			action.set(1.0f, 1);
			break;
		case "w":
			action.set(1.0f, 2);
			break;
		case "s":
			action.set(1.0f, 3);
			break;
		case "q":
			action.set(1.0f, 4);
			break;
		case "e":
			action.set(1.0f, 5);
			break;
		case "Enter":
			action.set(1.0f, 6);
			break;
		}
		
		controller.setAction(action);
	}
	
}
