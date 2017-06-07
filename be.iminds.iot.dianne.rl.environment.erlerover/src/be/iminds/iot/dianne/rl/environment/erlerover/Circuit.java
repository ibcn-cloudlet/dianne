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
package be.iminds.iot.dianne.rl.environment.erlerover;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import be.iminds.iot.simulator.api.Orientation;
import be.iminds.iot.simulator.api.Position;
import be.iminds.iot.simulator.api.Simulator;

public class Circuit {

	public String name;
	
	public List<float[]> spawnPoses = new ArrayList<>();
	
	public Circuit(String name, float... spawnPoses){
		this.name = name;
		
		for(int i=0;i<spawnPoses.length;i+=6){
			float[] pose = Arrays.copyOfRange(spawnPoses, i, i+6);
			this.spawnPoses.add(pose);
		}
	}
	
	public void spawn(Simulator sim){
		int spawn = (int)(Math.random() * spawnPoses.size());
		float[] spawnPose = spawnPoses.get(spawn);
		
		try {
			sim.getPosition(name+".sdf");
		} catch(Exception e){
			// first move rover out of the way?
			//sim.setPosition("rover", new Position(-10, -10, 0));
			sim.loadScene("scenes/"+name+".sdf");
		}
		sim.setPosition("rover", new Position(spawnPose[0], spawnPose[1], spawnPose[2]));
		sim.setOrientation("rover", new Orientation(spawnPose[3], spawnPose[4], spawnPose[5]));
		try {
			sim.getPosition(name+".sdf");
		} catch(Exception e){
			System.out.println("Failed to instantiate circuit "+name+", try again?!");
			spawn(sim);
		}
	}
}
