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
package be.iminds.iot.dianne.rl.learn.strategy;

import be.iminds.iot.dianne.api.nn.learn.LearnProgress;

/**
 * Represents the progress made by a State Belief Learner
 * 
 * @author tverbele
 *
 */
public class StateBeliefLearnProgress extends LearnProgress{

	public final float priorRegulLoss;
	public final float posteriorRegulLoss;
	public final float observationReconLoss;
	public final float rewardReconLoss;
	
	public StateBeliefLearnProgress(long iteration, 
			float priorRegulLoss, 
			float posteriorRegulLoss, 
			float observationReconLoss,
			float rewardReconLoss){
		super(iteration, priorRegulLoss+posteriorRegulLoss+observationReconLoss+rewardReconLoss);
		this.priorRegulLoss = priorRegulLoss;
		this.posteriorRegulLoss = posteriorRegulLoss;
		this.observationReconLoss = observationReconLoss;
		this.rewardReconLoss = rewardReconLoss;
	}
	
	@Override
	public String toString(){
		return "[LEARNER] Iteration: "+iteration+" Loss: "+minibatchLoss;
	}
}
