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
package be.iminds.iot.dianne.rl.environment.kuka;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.environment.kuka.api.KukaEnvironment;
import be.iminds.iot.dianne.rl.environment.kuka.config.FetchCanContinuousConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.robot.api.JointDescription;


@Component(immediate = true,
	service = {Environment.class, KukaEnvironment.class},
	property = { "name="+GrabCanContinuousEnvironment.NAME, 
				 "aiolos.unique=true",
				 "aiolos.combine=*",
				 "osgi.command.scope=grabcancont",
				 "osgi.command.function=start",
				 "osgi.command.function=stop",
				 "osgi.command.function=pause",
				 "osgi.command.function=resume",
				 "osgi.command.function=reward",
				 "osgi.command.function=resetCan"})
public class GrabCanContinuousEnvironment extends AbstractGrabCanEnvironment {
	
	public static final String NAME = "GrabCanContinuous";
	
	protected FetchCanContinuousConfig config;
	private float min = -1.0f;
	private float max = 1.0f;
	
	@Override
	public String getName(){
		return NAME;
	}

	@Override
	public int[] actionDims() {
		return new int[]{6};
	}
	
	@Override
	protected void executeAction(Tensor t) throws Exception {
		float[] f = Arrays.copyOf(t.get(), t.get().length + 1);
		f[f.length - 1] = f[f.length -2];
		
		List<JointDescription> joints = kukaArm.getJoints();
		float a,b;
		JointDescription joint;
		for (int i=0; i < f.length; i++) {
			joint = joints.get(i);
			a = joint.getPositionMin();
			b = joint.getPositionMax();
			f[i] = (((b-a)*(f[i]-min))/(max - min)) + a; // tranform to new range.
		}
		kukaArm.setPositions(f);
	
		// simulate an iteration further
		if(simulator != null && super.config.tick){
			for(int i=0;i<=super.config.skip;i++){
				simulator.tick();
			}
		}
	}
	
	@Override
	protected float calculateEnergy(Tensor a) {
		return 0;
	}
	
	@Override
	public void configure(Map<String, String> config) {
		this.config = DianneConfigHandler.getConfig(config, FetchCanContinuousConfig.class);
		
		super.configure(config);
	}
}
