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
package be.iminds.iot.dianne.rnn.learn.sampling;

/**
 * Strategy to select a next index to visit in the learning procedure.
 * 
 * @author tverbele
 *
 */
public interface SequenceSamplingStrategy {

	/**
	 * @return new sequence to sample from
	 */
	int sequence();
	
	/**
	 * @param sequence the sequence of which to sample
	 * @param length the length of samples that are requested from the start index
	 * @return next start index of sequence
	 */
	int next(int sequence, int length);
	
	/**
	 * @param count number of sequene indices to generate
	 * @return count sequence indices of the dataset to visit
	 */
	default int[] sequence(int count){
		int[] indices = new int[count];
		for(int i=0;i<count;i++){
			indices[i] = sequence();
		}
		return indices;
	}
	
	/**
	 * get a start index for each sequence index in the array
	 * @param sequence the sequences of which to sample
	 * @param length the length of samples that are requested from the start index
	 * @return
	 */
	default int[] next(int[] sequence, int length){
		int[] indices = new int[sequence.length];
		for(int i=0;i<indices.length;i++){
			indices[i] = next(sequence[i], length);
		}
		return indices;
	}
}
