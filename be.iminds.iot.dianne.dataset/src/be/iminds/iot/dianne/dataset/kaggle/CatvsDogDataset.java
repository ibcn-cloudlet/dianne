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
import java.util.List;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.dianne.tensor.util.ImageConverter;

@Component(immediate = true, property = {"name=Kaggle (CatvsDog)","aiolos.unique=true" })
public class CatvsDogDataset implements Dataset {

	private TensorFactory factory;
	private ImageConverter converter;
	
	private List<Sample> data = new ArrayList<Sample>();
	private String[] labels;
	private int[] outputs;
	
	private int noSamples;
	
	private String dir = "";
	private String[] files;
	
	@Reference
	void setTensorFactory(TensorFactory f) {
		this.factory = f;
		this.converter = new ImageConverter(f);
	}
	
	@Activate
	public void activate(BundleContext context) {
		String d = context
				.getProperty("be.iminds.iot.dianne.dataset.kaggle.location");
		if (d != null) {
			this.dir = d;
		}
	
		File images = new File(dir + "train/");
		files = images.list();
		noSamples = files.length;
		
	}
	
	@Override
	public int size() {
		return noSamples;
	}

	@Override
	public Tensor getInputSample(int index) {
		String file = files[index];
		Tensor t = null;
		try {
			t = converter.readFromFile(file);
		} catch(Exception e){
			e.printStackTrace();
			System.out.println("Failed to load input sample "+file);
		}
		return t;
	}

	@Override
	public Tensor getOutputSample(int index) {
		Tensor output = factory.createTensor(3);
		output.fill(0.0f);
		String file = files[index];
		if(file.startsWith("cat")){
			output.set(1.0f, 0);
		} else if(file.startsWith("dog")){
			output.set(1.0f, 1);
		} else {
			output.set(1.0f, 2);
		}
		return output;
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
