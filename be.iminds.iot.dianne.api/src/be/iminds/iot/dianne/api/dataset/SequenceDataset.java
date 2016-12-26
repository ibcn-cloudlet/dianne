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
package be.iminds.iot.dianne.api.dataset;

/**
 * A SequenceDataset is a Dataset consisting of sequences of samples.
 * 
 * A sequence can for example be a time-series, used for prediction tasks where one
 * has to predict samples coming next in the sequence. In this case the target of 
 * sample s1 will be the input of sample s2 (possibly the same Tensor object).
 * 
 * @author tverbele
 *
 */
public interface SequenceDataset<S extends Sample, B extends Batch> extends Dataset {
	
	/**
	 * Number of sequences in the dataset
	 * @return number of sequences in the dataset
	 */
	public int sequences();
	
	/**
	 * Get a (part of) a sequence with start index and length
	 * @param s provided array to copy the data into, will be created in case of null or elements will be added if s.size() < length
	 * @param sequence number of the sequence
	 * @param index start index of the first sample
	 * @param length number of samples to get (in case sequence is not long enough, elements will be put to NaN)
	 * @return sequence
	 */
	Sequence<S> getSequence(Sequence<S> s, final int sequence, final int index, final int length);
	
	default Sequence<S> getSequence(final int sequence, final int index, final int length){
		return getSequence(null, sequence, index, length);
	}
	
	default Sequence<S> getSequence(final int sequence){
		return getSequence(null, sequence, 0, -1);
	}
	
	/**
	 * Get a (part of) sequences in batch with start indices and length
	 * @param s provided array to copy the data into, will be created in case of null or elements will be added if s.size() < length
	 * @param sequences sequences to batch
	 * @param indices start indices of the first samples
	 * @param length number of samples to get (in case sequence is not long enough, elements will be put to NaN)
	 * @return sequence
	 */
	Sequence<B> getBatchedSequence(Sequence<B> b, final int[] sequences, final int[] indices, final int length);
	
	default Sequence<B> getBatchedSequence(final int[] sequences, final int[] indices, final int length){
		return getBatchedSequence(null, sequences, indices, length);
	}
	
	default Sequence<B> getBatchedSequence(final int[] sequences){
		return getBatchedSequence(null, sequences, null, -1);
	}
}
