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

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeoutException;

import org.osgi.service.component.annotations.Component;
import org.osgi.util.promise.Promise;

import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.rl.environment.kuka.api.KukaEnvironment;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.robot.api.JointDescription;
import be.iminds.iot.robot.api.JointState;
import be.iminds.iot.robot.api.arm.Arm;
import be.iminds.iot.simulator.api.Position;


@Component(immediate = true,
	service = {Environment.class, KukaEnvironment.class},
	property = { "name="+ReacherGoalEnvironment.NAME, 
				 "aiolos.unique=true",
				 "aiolos.combine=*",
				 "osgi.command.scope=reachergoal",
				 "osgi.command.function=start",
				 "osgi.command.function=stop",
				 "osgi.command.function=pause",
				 "osgi.command.function=resume",
				 "osgi.command.function=reward"})
public class ReacherGoalEnvironment extends AbstractKukaEnvironment {
	
	public static final String NAME = "ReacherGoal";
	
	private float min = -1.0f;
	private float max = 1.0f;
	private float threshold = 0.01f;
	
	private int stateSize = 10;
	private int goalSize = 3;
	private Tensor goal;
	private Tensor start;
	
	@Override
	public String getName(){
		return NAME;
	}

	@Override
	public int[] actionDims() {
		return new int[]{stateSize/2};
	}
	
	@Override
	public int[] observationDims() {
		return new int[]{this.kukaArm.getState().size()+2*goalSize};
	}
	
	@Override
	protected void updateObservation(){
		// observation = state (positions), goal, "current goal"
		
		// state = joint positions
		float[] result = new float[stateSize+2*goalSize];
		JointDescription joint = null;
		JointState state = null;
		float minPos, maxPos, minVel, maxVel;
		for (int i=0; i<stateSize/2; i++) {
			state = this.kukaArm.getState().get(i);
			joint = this.kukaArm.getJoints().get(i);
			minPos = joint.getPositionMin();
			maxPos = joint.getPositionMax();
			result[i] = 2*(state.position - minPos)/(maxPos - minPos)-1;
			
			minVel = joint.getVelocityMin();
			maxVel = joint.getVelocityMax();
			result[stateSize/2+i] = 2*(state.velocity - minVel)/(maxVel - minVel)-1;
		}
		
		// environment goal
		float[] g = goal.get();
		result[stateSize] = g[0];
		result[stateSize+1] = g[1];
		result[stateSize+2] = g[2];
		
		System.arraycopy(goal.get(), 0, result, stateSize, goalSize);
		
		// currently "reached" goal
		Position p = simulator.getPosition("youBot_positionTip", "arm_ref");
		result[stateSize+goalSize] = p.x;
		result[stateSize+goalSize+1] = p.y;
		result[stateSize+goalSize+2] = p.z;
		
		observation.set(result);
	}
	
	@Override
	protected void executeAction(Tensor t) throws Exception {
		// apply velocities
		float[] f = t.get();
		
		List<JointDescription> joints = kukaArm.getJoints();
		float a,b;
		JointDescription joint;
		for (int i=0; i < f.length; i++) {
			joint = joints.get(i);
			a = joint.getVelocityMin();
			b = joint.getVelocityMax();
			
			f[i] = (f[i]-min)/(max - min)*(b-a) + a; // tranform from [min,max] to new range.
		}
		kukaArm.setVelocities(f);
	
		// simulate an iteration further
		if(simulator != null){
			simulator.tick();
		}
	}
	
	@Override
	protected boolean checkCollisions(){
		return false; // don't worry about collisions...
	}
	
	@Override
	protected void initAction(){
		// reset arm to candle
		Promise<Arm> p = kukaArm.moveTo(start.get(0), start.get(1), start.get(2));
		int i = 0;
		while(!p.isDone() && active && i < 1000) {
			try {
				simulator.tick();
				i++;
			} catch(TimeoutException e){}
		} 
	}

	@Override
	protected void configure(Map<String, String> config) {
		// hacky way to disable front facing hokuyo sensor
		super.config.simState = true;
		super.config.simReset = 20; // reset each 20 episodes to avoid memory leaking V-REP with Newton dynamics
		super.config.scene = "youbot_reacher.ttt";
		
		start = new Tensor(new float[]{0.4f, 0.0f, 0.4f}, 3);
		goal = new Tensor(goalSize);
		
		observation = new Tensor(stateSize+2*goalSize);
		
		if(config.containsKey("threshold")){
			this.threshold = Float.parseFloat(config.get("threshold"));
		}
	}

	@Override
	protected float calculateReward() throws Exception {
		Position p = simulator.getPosition("youBot_positionTip", "arm_ref");
		
		Tensor current = new Tensor(new float[]{p.x, p.y, p.z},3);
		if(current.equals(goal, threshold)){
			terminal = true;
			return 0;
		}
		
		return -1;
	}

	@Override
	protected void resetEnvironment() throws Exception {
		
		double d = Math.random()*0.3+0.24;
		double a = (Math.random()-0.5)*Math.PI;
		double z = Math.random()*0.3;
		
		float[] g = new float[3];
		g[0] = (float) (d*Math.cos(a));
		g[1] = (float) (d*Math.sin(a));
		g[2] = (float) z;
		goal.set(g);

		simulator.setPosition("target", "arm_ref", new Position(g[0],g[1],g[2]));
				
	}
}
