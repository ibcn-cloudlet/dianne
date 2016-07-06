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


import java.io.File;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;

/**
 * A generic ImageSegmentationDataset that contains a number of numbered .jpg files
 * in a images/ subdir, together with corresponding masks in a masks/ dir
 * 
 * @author tverbele
 *
 */
@Component(
		service={Dataset.class},
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.ImageSegmentationDataset",
		immediate=true, 
		property={"aiolos.unique=true"})
public class GenericImageSegmentationDataset extends ImageSegmentationDataset {

	@Override
	protected String getImageFile(int index) {
		return dir+File.separator+ "images/" + index + ".jpg";
	}

	@Override
	protected String getMaskFile(int index) {
		return dir+File.separator+ "masks/" + index + ".jpg";
	}

	@Override
	protected void init(Map<String, Object> properties) {
		File images = new File(properties.get("dir") + File.separator + "images/");
		this.noSamples = images.list().length;
	}

	@Override
	protected void readLabels(String labelsFile) {
		// no labels in segmentation dataset
	}
	
}