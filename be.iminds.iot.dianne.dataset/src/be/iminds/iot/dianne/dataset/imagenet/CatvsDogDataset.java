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
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

/**
 * Exposes the ImageNet dataset as Cats and Dogs
 * 
 * @author tverbele
 *
 */
@Component(immediate = true, property = {"name=ImageNet (CatvsDog)","aiolos.callback=be.iminds.iot.dianne.api.dataset.Dataset" })
public class CatvsDogDataset implements Dataset {

	private Dataset imagenet;
	private TensorFactory factory;
	
	// binary array of the ImageNet labels that map to cat/dog
	private boolean[] cats;
	private boolean[] dogs;

	private String dir = "";
	
	@Reference(target="(name=ImageNet)")
	void setDataset(Dataset d) {
		this.imagenet = d;
	}

	@Reference
	void setTensorFactory(TensorFactory f) {
		this.factory = f;
	}
	
	@Activate
	public void activate(BundleContext context) {
		String d = context
				.getProperty("be.iminds.iot.dianne.dataset.imagenet.location");
		if (d != null) {
			this.dir = d;
		}
		
		List<String> catLabels = readStrings(dir +"cats.txt");
		List<String> dogLabels = readStrings(dir +"dogs.txt");
		
		String[] imageNetLabels = imagenet.getLabels();
		cats = new boolean[imageNetLabels.length];
		dogs = new boolean[imageNetLabels.length];
		for(int i=0;i<imageNetLabels.length;i++){
			if(catLabels.contains(imageNetLabels[i])){
				cats[i] = true;
			} else {
				cats[i] = false;
			}
			
			if(dogLabels.contains(imageNetLabels[i])){
				dogs[i] = true;
			} else {
				dogs[i] = false;
			}
		}
	}
	
	private List<String> readStrings(String file){
		List<String> result = new ArrayList<>();
		try {
			InputStream labelInput = new FileInputStream(file);

			ArrayList<String> l = new ArrayList<String>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					labelInput));
			String s;
			while ((s = reader.readLine()) != null) {
				String[] split = s.split(",");
				for(String part : split){
					result.add(part.trim());
				}
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return result;
	}
	
	@Override
	public String getName() {
		return "ImageNet (CatvsDog)";
	}

	@Override
	public int size() {
		return imagenet.size();
	}

	@Override
	public Tensor getInputSample(int index) {
		return imagenet.getInputSample(index);
	}

	@Override
	public Tensor getOutputSample(int index) {
		Tensor output = factory.createTensor(3);
		output.fill(0.0f);
		
		Tensor imagenetOutput = imagenet.getOutputSample(index);
		int max = factory.getTensorMath().argmax(imagenetOutput);
		
		if(cats[max]){
			output.set(1.0f, 0);
		} else if(dogs[max]){
			output.set(1.0f, 1);
		} else {
			output.set(1.0f, 2);
		}
		return output;
	}

	@Override
	public String[] getLabels() {
		return new String[]{"Cat","Dog","Other"};
	}
}
