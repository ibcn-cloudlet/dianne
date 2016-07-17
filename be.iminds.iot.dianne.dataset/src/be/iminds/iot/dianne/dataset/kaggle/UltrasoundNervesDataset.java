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
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.dataset.ImageSegmentationDataset;

/**
 * Kaggle ultrasound nerve segmentation task dataset
 * 
 * https://www.kaggle.com/c/ultrasound-nerve-segmentation/
 * 
 * @author tverbele
 *
 */
@Component(
		service={Dataset.class},
		immediate=true, 
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.UltrasoundNerves")
public class UltrasoundNervesDataset extends ImageSegmentationDataset {

	@Override
	protected String getImageFile(int index) {
		int i = index / 120 + 1;
		int j = index % 120 + 1;
		return dir+File.separator+ "train" + File.separator
				 + i +"_"+j+".tif";
	}

	@Override
	protected String getMaskFile(int index) {
		int i = index / 120 + 1;
		int j = index % 120 + 1;
		return dir+File.separator+ "train" + File.separator
				 + i +"_"+j+"_mask.tif";
	}

	@Override
	protected void init(Map<String, Object> properties) {
		this.name = "UltrasoundNerves";
		this.inputDims = new int[]{1,420,580};
		this.targetDims = new int[]{1,420,580};
		this.noSamples = 47*120;
	}
}
