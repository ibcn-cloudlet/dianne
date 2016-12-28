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
package be.iminds.iot.dianne.rnn.dataset.chars;

import java.io.File;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.AbstractDataset;
import be.iminds.iot.dianne.api.dataset.Batch;
import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.api.dataset.Sequence;
import be.iminds.iot.dianne.api.dataset.SequenceDataset;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * A Character SequenceDataset ... read a text file as a single sequence
 * 
 * @author tverbele
 *
 */
@Component(
		service={SequenceDataset.class, Dataset.class},
		immediate=true,
		property={"aiolos.unique=true","aiolos.combine=*"},
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.CharSequenceDataset")
public class CharSequenceDataset extends AbstractDataset implements SequenceDataset<Sample, Batch> {

	private String data;
	private String chars = "";

	@Override
	protected void init(Map<String, Object> properties) {
		try {
			inputType = "character";
			targetType = "characer";
			
			String file = "input.txt";
			
			if(properties.containsKey("file")){
				file = properties.get("file").toString();
			}
			
			// read the data
			byte[] encoded = Files.readAllBytes(Paths.get(dir+File.separator+file));
			data = new String(encoded, Charset.defaultCharset());
			
			noSamples = data.length();
			
			// no labels given, build up the vocabulary
			if(!properties.containsKey("labels") 
					&& !properties.containsKey("labelsFile")){
				
				data.chars().forEach(c -> {
					if(!chars.contains(""+(char)c)){
						chars+=""+(char)c;
					}
				});
				
				labels = new String[chars.length()];
				for(int i=0;i<labels.length;i++){
					labels[i] = ""+chars.charAt(i);
				}
				
				inputDims = new int[]{chars.length()};
				targetDims = new int[]{chars.length()};
			}
			
		} catch(Exception e){
			e.printStackTrace();
			throw new RuntimeException("Failed to load char sequence dataset", e);
		}	
	}

	@Override
	protected void readLabels(String labelsFile) {
		try {
			byte[] encoded = Files.readAllBytes(Paths.get(dir+File.separator+labelsFile));
			chars = new String(encoded, Charset.defaultCharset());
			
			labels = new String[chars.length()];
			for(int i=0;i<labels.length;i++){
				labels[i] = ""+chars.charAt(i);
			}
			
			inputDims = new int[]{chars.length()};
			targetDims = new int[]{chars.length()};
			
		} catch(Exception e){
			e.printStackTrace();
			throw new RuntimeException("Failed to load char sequence dataset", e);
		}	
	}

	@Override
	protected Tensor getInputSample(Tensor t, int index) {
		return asTensor(data.charAt(index), t);
	}

	@Override
	protected Tensor getTargetSample(Tensor t, int index) {
		return asTensor(data.charAt(index+1), t);
	}

	private Tensor asTensor(char c, Tensor t){
		int index = 0;
		index = chars.indexOf(c);
		if(t == null)
			t = new Tensor(chars.length());
		t.fill(0.0f);
		if(index == -1){
			System.err.println("Character "+c+" is not in the vocabulary");
			return t;
		}
		t.set(1.0f, index);
		return t;
	}
	
	private char asChar(Tensor t){
		int index = TensorOps.argmax(t);
		return chars.charAt(index);
	}

	@Override
	public int sequences() {
		return 1;
	}
	
	@Override
	public int sequenceLength(int index){
		if(index > 1){
			throw new ArrayIndexOutOfBoundsException();
		}
		return data.length();
	}

	@Override
	public Sequence<Sample> getSequence(Sequence<Sample> seq, int sequence, int index, int length) {
		if(seq == null){
			seq = new Sequence<Sample>();
		}
		List<Sample> s = seq.data;
		
		if(sequence > 1){
			throw new RuntimeException("Invalid sequence number");
		}

		if(index >= data.length()){
			throw new RuntimeException("Invalid start index: "+index);
		}
		
		if(length == -1){
			length = data.length();
		}
		
		Sample previous = null;
		for(int i=0;i<length;i++){
			Sample sample;
			if(s.size() <= i){
				sample = new Sample(previous != null ? previous.target : new Tensor(chars.length()), new Tensor(chars.length()));
				s.add(sample);
			} else {
				sample = s.get(i);
				if(previous != null){
					sample.input = previous.target;
				}
			}
			
			int k = index + i;
			if(i == 0){
				char c = data.charAt(k);
				asTensor(c, sample.input);
			} 
			
			if(k+1 < data.length()){
				char c = data.charAt(k+1);
				asTensor(c, sample.target);
			} else {
				sample.target.fill(Float.NaN);
			}
			
			previous = sample;
		}
		
		seq.size = length;
		return seq;
	}

	@Override
	public Sequence<Batch> getBatchedSequence(Sequence<Batch> seq, int[] sequences, int[] indices, int length) {
		if(seq == null){
			seq = new Sequence<Batch>();
		}
		List<Batch> b = seq.data;
		
		for(int sequence : sequences){
			if(sequence > 1){
				throw new RuntimeException("Invalid sequence number");
			}
		}

		if(indices != null){
			for(int index : indices){
				if(index >= data.length()){
					throw new RuntimeException("Invalid start index");
				}
			}
		}
		
		if(length == -1){
			length = data.length();
		}
		
		Batch previous = null;
		for(int i=0;i<length;i++){
			Batch batch;
			if(b.size() <= i){
				batch = new Batch(previous != null ? previous.target : new Tensor(sequences.length, chars.length()), new Tensor(sequences.length, chars.length()));
				b.add(batch);
			} else {
				batch = b.get(i);
				if(previous != null){
					batch.input = previous.target;
				}
			}
			
			for(int s=0;s<sequences.length;s++){
				int start = indices == null ? 0 : indices[s];
				int k = start + i;
				
				if(i == 0){
					char c = data.charAt(k);
					asTensor(c, batch.getSample(s).input);
				} 
				
				if(k+1 < data.length()){
					char c = data.charAt(k+1);
					asTensor(c, batch.getSample(s).target);
				} else {
					batch.getSample(s).target.fill(Float.NaN);
				}
			}
			
			previous = batch;
		}
		
		seq.size = length;
		return seq;
	}
}
