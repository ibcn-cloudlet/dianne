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
 *     Tim Verbelen, Steven Bohez, Elias De Coninck
 *******************************************************************************/
package be.iminds.iot.dianne.dataset.imagenet.validation;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.dataset.ImageClassificationDataset;

/**
 * Sample of the Imagenet dataset
 * 
 * Currently based on the ILSVRC 2012 validation set
 * 
 * @author tverbele, ejodconi
 *
 */
@Component(
		service={Dataset.class},
		immediate=true, 
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.ImageNet.validation")
public class ImageNetValidationDataset extends ImageClassificationDataset {

	@Override
	protected void init(Map<String, Object> properties) {
		this.name = "ImageNetValidation";
		this.inputDims = null;
		this.targetDims = new int[]{1000};

		File images = new File(properties.get("dir") + File.separator + "images/");
		noSamples = (int) Arrays.stream(images.listFiles()).filter(f -> !f.isHidden()).count();
		
		this.labelsFile = "classes.txt";
		this.targetsFile = "outputs.txt";
	}

	@Override
	protected String getImageFile(int index) {
		return dir+File.separator+ "images/" + "ILSVRC2012_val_"
				 + String.format("%08d", index+1) + ".JPEG";
	}

	@Override
	protected void readLabels(String file) {
		try {
			InputStream labelInput = new FileInputStream(dir + File.separator + file);

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

	@Override
	protected void readTargets(String file) {
		try {
			InputStream targetsInput = new FileInputStream(dir + File.separator + file);

			int i = 0;
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					targetsInput));
			String s;
			while ((s = reader.readLine()) != null) {
				int clazz = Integer.parseInt(s) - 1; // in the file this ranges
														// from 1..1000, convert
														// to 0..999
				targets[i++] = clazz;
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
