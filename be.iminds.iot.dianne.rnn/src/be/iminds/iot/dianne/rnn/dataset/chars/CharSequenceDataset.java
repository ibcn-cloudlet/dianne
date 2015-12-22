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

import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.rnn.SequenceDataset;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

/**
 * A Character SequenceDataset ... read a text file and treats each character as input/output sample
 * 
 * @author tverbele
 *
 */
@Component(immediate=true, property={"name=CharSequence","aiolos.unique=true"})
public class CharSequenceDataset implements SequenceDataset{

	private TensorFactory factory;
	
	private String file = "input.txt";
	
	private String data;
	private String chars = "";
	private String[] labels;
	
	@Reference
	void setTensorFactory(TensorFactory f){
		this.factory = f;
	}
	
	@Activate
	public void activate(BundleContext context) throws Exception {
		String f = context.getProperty("be.iminds.iot.dianne.dataset.chars.location");
		if(f!=null){
			this.file = f;
		}
		
		try {
			byte[] encoded = Files.readAllBytes(Paths.get(file));
			data = new String(encoded, Charset.defaultCharset());
			
			data.chars().forEach(c -> {
				if(!chars.contains(""+(char)c)){
					chars+=""+(char)c;
				}
			});
			
			labels = new String[chars.length()];
			for(int i=0;i<labels.length;i++){
				labels[i] = ""+chars.charAt(i);
			}
			
		} catch(Exception e){
			System.err.println("Failed to load char sequence dataset ... ");
			throw e;
		}
	}

	@Override
	public int size() {
		return data.length();
	}

	@Override
	public Tensor getInputSample(int index) {
		return asTensor(data.charAt(index));
	}

	@Override
	public Tensor getOutputSample(int index) {
		return asTensor(data.charAt(index+1));
	}

	@Override
	public Tensor[] getSequence(int index, int length) {
		Tensor[] sequence = new Tensor[length+1];
		for(int i=index;i<index+length+1;i++){
			sequence[i-index] = asTensor(data.charAt(i));
		}
		return sequence;
	}

	@Override
	public String getName() {
		return "CharSequence";
	}

	@Override
	public String[] getLabels() {
		return labels;
	}
	
	private Tensor asTensor(char c){
		int index = chars.indexOf(c);
		Tensor t = factory.createTensor(chars.length());
		t.fill(0.0f);
		t.set(1.0f, index);
		return t;
	}
	
	private char asChar(Tensor t){
		int index = factory.getTensorMath().argmax(t);
		return chars.charAt(index);
	}
}
