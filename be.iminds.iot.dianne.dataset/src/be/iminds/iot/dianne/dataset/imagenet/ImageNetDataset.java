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
package be.iminds.iot.dianne.dataset.imagenet;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
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

/**
 * Sample of the Imagenet dataset
 * 
 * Currently based on the ILSVRC 2012 validation set
 * 
 * @author tverbele
 *
 */
@Component(immediate = true, property = {"name=ImageNet","aiolos.unique=true" })
public class ImageNetDataset implements Dataset {

	private TensorFactory factory;
	private ImageConverter converter;

	private List<Sample> data = new ArrayList<Sample>();
	private String[] labels;
	private int[] outputs;

	private int outputSize;
	private int noSamples;

	private String dir = "";

	@Reference
	void setTensorFactory(TensorFactory f) {
		this.factory = f;
		this.converter = new ImageConverter(f);
	}

	@Activate
	public void activate(BundleContext context) {
		String d = context
				.getProperty("be.iminds.iot.dianne.dataset.imagenet.location");
		if (d != null) {
			this.dir = d;
		}

		outputSize = 1000;

		File images = new File(dir + "images/");
		noSamples = (int) Arrays.stream(images.listFiles()).filter(f -> !f.isHidden()).count();
		
		readLabels("classes.txt");
		readOutputs("outputs.txt");
	}

	private void readLabels(String file) {
		try {
			InputStream labelInput = new FileInputStream(dir + file);

			ArrayList<String> l = new ArrayList<String>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					labelInput));
			String s;
			while ((s = reader.readLine()) != null) {
				// only pick first label in case of multiple definitions
				int comma = s.indexOf(",");
				if (comma > 0) {
					s = s.substring(0, comma);
				}
				l.add(s);
			}
			labels = new String[l.size()];
			for (int i = 0; i < l.size(); i++) {
				labels[i] = l.get(i);
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void readOutputs(String file) {
		try {
			InputStream outputsInput = new FileInputStream(dir + file);

			ArrayList<Integer> l = new ArrayList<Integer>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					outputsInput));
			String s;
			while ((s = reader.readLine()) != null) {
				// only pick first label in case of multiple definitions
				int clazz = Integer.parseInt(s) - 1; // in the file this ranges
														// from 1..1000, convert
														// to 0..999
				l.add(clazz);
			}
			// TODO this assumes one only has the first x samples...
			outputs = new int[noSamples];
			for (int i = 0; i < noSamples; i++) {
				outputs[i] = l.get(i);
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public String getName() {
		return "ImageNet";
	}

	@Override
	public int size() {
		return noSamples;
	}
	
	@Override
	public Tensor getInputSample(int index) {
		// Open JPEG file and convert to size
		String file = dir + "images/" + "ILSVRC2012_val_"
				+ String.format("%08d", index+1) + ".JPEG";
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
		Tensor output = factory.createTensor(outputSize);
		output.fill(0.0f);
		output.set(1.0f, outputs[index]);
		return output;
	}

	@Override
	public String[] getLabels() {
		return labels;
	}


}
