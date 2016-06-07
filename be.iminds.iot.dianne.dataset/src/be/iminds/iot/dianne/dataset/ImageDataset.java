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
package be.iminds.iot.dianne.dataset;

import java.util.Map;

import be.iminds.iot.dianne.api.dataset.AbstractDataset;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.util.ImageConverter;

/**
 * This provides an abstract class to implement Datasets that have images as input.
 * The images are stored on the filesystem as image files, with a separate file
 * containing an output class index and the labels. 
 *  
 * @author tverbele
 *
 */
public abstract class ImageDataset extends AbstractDataset {

	protected ImageConverter converter = new ImageConverter();

	protected String outputsFile;
	protected int[] outputs;

	@Override
	protected void activate(Map<String, Object> properties) {
		super.activate(properties);

		if(properties.containsKey("outputsFile")){
			outputsFile = (String)properties.get("outputsFile");
		}

		outputs = new int[noSamples];
		readOutputs(outputsFile);
	}

	protected abstract String getImageFile(int index);
	
	protected abstract void readOutputs(String file);
	
	@Override
	public Tensor getInputSample(Tensor t, int index) {
		// Open JPEG file and convert to size
		String file = getImageFile(index);
		try {
			t = converter.readFromFile(file, t);
		} catch(Exception e){
			e.printStackTrace();
			System.out.println("Failed to load input sample "+file);
		}
		return t;
	}

	@Override
	public Tensor getOutputSample(Tensor t, int index) {
		if(t == null)
			t = new Tensor(outputSize);
		t.fill(0.0f);
		t.set(1.0f, outputs[index]);
		return t;
	}

}
