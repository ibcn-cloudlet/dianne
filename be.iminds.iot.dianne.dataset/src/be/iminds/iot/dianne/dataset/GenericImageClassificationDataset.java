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


import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;

/**
 * A generic ImageClassificationDataset that contains a number of numbered .jpg files
 * in a images/ subdir, together with a labels and targets file.
 * The labels file contains a label on each line, the targets file
 * contains an class index on each line.
 * 
 * @author tverbele
 *
 */
@Component(
		service={Dataset.class},
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.ImageClassificationDataset",
		immediate=true, 
		property={"aiolos.unique=true"})
public class GenericImageClassificationDataset extends ImageClassificationDataset {

	@Override
	protected String getImageFile(int index) {
		return dir+File.separator+ "images/" + index + ".jpg";
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
				int clazz = Integer.parseInt(s);
				targets[i++] = clazz;
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}	
	}

	@Override
	protected void init(Map<String, Object> properties) {
		File images = new File(dir + File.separator + "images/");
		this.noSamples = images.list().length;
	}

	@Override
	protected void readLabels(String labelsFile) {
		try {
			InputStream labelInput = new FileInputStream(dir + File.separator + labelsFile);

			ArrayList<String> l = new ArrayList<String>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					labelInput));
			String s;
			while ((s = reader.readLine()) != null) {
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
}