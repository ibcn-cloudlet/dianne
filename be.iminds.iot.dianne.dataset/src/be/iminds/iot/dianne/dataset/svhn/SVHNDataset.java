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
package be.iminds.iot.dianne.dataset.svhn;

import java.io.InputStream;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.dataset.FileDataset;

/**
 * The Street View House Numbers dataset, based on the cropped 32x32 images
 * http://ufldl.stanford.edu/housenumbers/
 * 
 * @author tverbele
 *
 */
@Component(
		service={Dataset.class},
		immediate=true, 
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.SVHN")
public class SVHNDataset extends FileDataset{

	@Override
	protected void init(Map<String, Object> properties){
		this.name = "SVHN";
		this.inputDims = new int[]{3, 32, 32};
		this.outputDims = new int[]{10};
		this.noSamples = 73257+26032;
		this.labels = new String[]{"0","1","2","3","4","5","6","7","8","9"};

		this.inputFiles = new String[]{"train_images.bin", "test_images.bin"};
		this.outputFiles = new String[]{"train_labels.bin", "test_labels.bin"};
	}
	
	@Override
	protected void parse(InputStream in, InputStream out) throws Exception{
		while(in.available()>0){
			int i = readUByte(out);
			outputs[count][i] = 1;
			
			for(int j=0;j<inputSize;j++){
				inputs[count][j] = (float)readUByte(in)/255f;
			}
			
			count++;
		}
	}
}
