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
 * Each image has to be classified as one of a discrete set of labeled classes.
 * The images are stored on the filesystem as image files, with a separate file
 * containing an target class index and the labels. 
 *  
 * @author tverbele
 *
 */
public abstract class ImageClassificationDataset extends AbstractDataset {

	protected ImageConverter converter = new ImageConverter();

	protected String targetsFile;
	protected int[] targets;

	@Override
	protected void activate(Map<String, Object> properties) {
		super.activate(properties);

		if(properties.containsKey("targetsFile")){
			targetsFile = (String)properties.get("targetsFile");
		}

		targets = new int[noSamples];
		readTargets(targetsFile);
	}

	protected abstract String getImageFile(int index);
	
	protected abstract void readTargets(String file);
	
	@Override
	public Tensor getInputSample(Tensor t, int index) {
		// Open JPEG file and convert to size
		String file = getImageFile(index);
		try {
			t = converter.fromFile(file, t);
		} catch(Exception e){
			e.printStackTrace();
			System.out.println("Failed to load input sample "+file);
		}
		return t;
	}

	@Override
	public Tensor getTargetSample(Tensor t, int index) {
		if(t == null)
			t = new Tensor(targetSize);
		t.fill(0.0f);
		t.set(1.0f, targets[index]);
		return t;
	}

}
