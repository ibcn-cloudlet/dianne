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
package be.iminds.iot.dianne.dataset.mnist;

import java.io.InputStream;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.dataset.FileDataset;

/**
 * The MNIST dataset, uses the images form LeCun's website:
 * http://yann.lecun.com/exdb/mnist/
 * 
 * @author tverbele
 *
 */
@Component(
		service={Dataset.class},
		immediate=true, 
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.MNIST")
public class MNISTDataset extends FileDataset {

	@Override
	protected void init(Map<String, Object> properties){
		this.name = "MNIST";
		this.inputDims = new int[]{1, 28, 28};
		this.outputDims = new int[]{10};
		this.noSamples = 70000;
		this.labels = new String[]{"0","1","2","3","4","5","6","7","8","9"};

		inputFiles = new String[]{"train-images.idx3-ubyte", "t10k-images.idx3-ubyte"};
		outputFiles = new String[]{"train-labels.idx1-ubyte", "t10k-labels.idx1-ubyte"};
	}
	
	@Override
	protected void parse(InputStream in, InputStream out) throws Exception {
		int magic = readInt(in);
		assert magic == 2051;
		int noImages = readInt(in);
		int noRows = readInt(in);
		int noColumns = readInt(in);
				
		magic = readInt(out);
		assert magic == 2049;
		int noLabels = readInt(out);
		
		for(int read = 0;read<noImages;read++){
			for(int j=0;j<inputSize;j++){
				inputs[count][j] = (float)readUByte(in)/255f;
			}
			                               
			int i = readUByte(out);
			outputs[count][i] = 1;
			
			count++;
		}
	}
	
}
