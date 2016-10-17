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
 *     Tim Verbelen
 *******************************************************************************/
package be.iminds.iot.dianne.nn.learn.strategy;

import be.iminds.iot.dianne.api.nn.learn.LearnProgress;

public class GenerativeAdverserialLearnProgress extends LearnProgress {

	public final float d_loss;
	public final float g_loss;
	
	public GenerativeAdverserialLearnProgress(long iteration, 
			float d_loss_pos, float d_loss_neg,
			float g_loss) {
		super(iteration, g_loss);
		this.d_loss = d_loss_pos+d_loss_neg;
		this.g_loss = g_loss;
	}

	@Override
	public String toString(){
		return "[LEARNER] Batch: "+iteration+" D Loss: "+d_loss+" G Loss: "+g_loss;
	}
}
