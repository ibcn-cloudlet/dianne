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
package be.iminds.iot.dianne.rl.module;

import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.Join;
import be.iminds.iot.dianne.tensor.ModuleOps;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class NormalizedAdvantageFunction extends Join {
	
	private final int actionDims;
	
	private Tensor eye;
	private Tensor lower;
	
	private Tensor L;
	private Tensor Lt;
	private Tensor P;
	
	private Tensor da;
	private Tensor Pda;
	
	private Tensor dPda;
	private Tensor dda;
	
	private Tensor dP;
	private Tensor dL;
	
	public NormalizedAdvantageFunction(int actionDims){
		super();
		this.actionDims = actionDims;
		init();
	}
	
	public NormalizedAdvantageFunction(UUID id, int actionDims){
		super(id);
		this.actionDims = actionDims;
		init();
	}
	
	protected void init() {
		eye = new Tensor(actionDims, actionDims);
		eye.fill(0);
		for(int d = 0; d < actionDims; d++) {
			eye.set(1, d, d);
		}
		
		lower = new Tensor(actionDims, actionDims);
		lower.fill(0);
		for(int r = 1; r < actionDims; r++) {
			for(int c = 0; c < r; c++) {
				lower.set(1, r, c);
			}
		}
		
		output = new Tensor(1);
	}
	
	@Override
	protected void forward() {
		if(prev==null || prevIds.length!=3){
			throw new RuntimeException("NormalizedAdvantageFunction not configured correctly, should receive a, mu and L");
		}
		
		Tensor action = inputs.get(prevIds[0]);
		Tensor maxAction = inputs.get(prevIds[1]);
		Tensor matrix = inputs.get(prevIds[2]);
		
		L = ModuleOps.softplus(L, matrix, 1, 20);
		TensorOps.cmul(L, eye, L);
		TensorOps.addcmul(L, L, 1, lower, matrix);
		
		Lt = L.transpose(Lt, 0, 1);
		P = TensorOps.mm(P, L, Lt);
		
		da = TensorOps.sub(da, action, maxAction);
		Pda = TensorOps.mv(Pda, P, da);
		output.fill(-TensorOps.dot(da, Pda)/2);
	}

	@Override
	protected void backward() {
		Tensor matrix = inputs.get(prevIds[2]);
		
		Tensor gradAction = gradInputs.computeIfAbsent(prevIds[0], k -> new Tensor(actionDims));
		Tensor gradMaxAction = gradInputs.computeIfAbsent(prevIds[1], k -> new Tensor(actionDims));
		Tensor gradMatrix = gradInputs.computeIfAbsent(prevIds[2], k -> new Tensor(actionDims, actionDims));
		
		float dout = gradOutput.get(0);
		
		dPda = TensorOps.mul(dPda, da, -dout/2);
		dda = TensorOps.tmv(dda, P, dPda);
		TensorOps.add(dda, dda, -dout/2, Pda);
		
		dP = TensorOps.vv(dP, dPda, da);
		dL = TensorOps.mm(dL, dP, Lt);
		TensorOps.mul(dL, dL, 2);
		
		dda.copyInto(gradAction);
		TensorOps.mul(gradMaxAction, dda, -1);
		
		ModuleOps.softplusGradIn(gradMatrix, dL, matrix, L, 1, 20);
		TensorOps.cmul(gradMatrix, eye, gradMatrix);
		TensorOps.addcmul(gradMatrix, gradMatrix, 1, lower, dL);
	}
	
}
