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
package be.iminds.iot.dianne.dataset.kaggle;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.util.ImageConverter;

@Component(immediate = true, property = {"name=Kaggle (CatvsDog)","aiolos.unique=true" })
public class CatvsDogDataset implements Dataset {

	private ImageConverter converter = new ImageConverter();
	
	private List<Sample> data = new ArrayList<Sample>();
	private String[] labels;
	private int[] outputs;
	
	private int noSamples;
	
	private String dir = "";
	private String[] files;
	
	@Activate
	public void activate(BundleContext context) {
		String d = context
				.getProperty("be.iminds.iot.dianne.dataset.kaggle.location");
		if (d != null) {
			this.dir = d;
		}
	
		File images = new File(dir + "train/");
		files = Arrays.stream(images.listFiles()).filter(f -> !f.isHidden()).map(f -> f.getName()).collect(Collectors.toList()).toArray(files);
		noSamples = files.length;
	}
	
	@Override
	public int[] inputDims(){
		return null;
	}
	
	@Override
	public int[] outputDims(){
		return new int[]{3};
	}
	
	@Override
	public int size() {
		return noSamples;
	}

	@Override
	public Tensor getInputSample(int index, Tensor t) {
		String file = files[index];
		try {
			t = converter.readFromFile(file, t);
		} catch(Exception e){
			e.printStackTrace();
			System.out.println("Failed to load input sample "+file);
		}
		return t;
	}

	@Override
	public Tensor getOutputSample(int index, Tensor t) {
		if(t == null)
			t = new Tensor(3);
		t.fill(0.0f);
		String file = files[index];
		if(file.startsWith("cat")){
			t.set(1.0f, 0);
		} else if(file.startsWith("dog")){
			t.set(1.0f, 1);
		} else {
			t.set(1.0f, 2);
		}
		return t;
	}

	@Override
	public String getName() {
		return "Kaggle (CatvsDog)";
	}

	@Override
	public String[] getLabels() {
		return new String[]{"Cat", "Dog", "Other"};
	}

}
