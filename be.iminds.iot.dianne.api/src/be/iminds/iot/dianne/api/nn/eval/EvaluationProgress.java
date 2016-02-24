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
package be.iminds.iot.dianne.api.nn.eval;

/**
 * Represents the progress made by an Evaluator
 * 
 * @author tverbele
 *
 */
public class EvaluationProgress extends Evaluation {

	/** Total number of samples to be processed */
	private final long total;
	private final long processed;
	
	public EvaluationProgress(long processed, long total, long time){
		super(null, null, null, time); // TODO actually give intermediate confusion matrix/outputs here?
		this.processed = processed;
		this.total = total;
	}
	
	/**
	 * @return the number of samples processed
	 */
	public long getProcessed(){
		return processed;
	}
	
	/**
	 * @return the total number of samples to be processed
	 */
	public long getTotal(){
		return total;
	}
}
