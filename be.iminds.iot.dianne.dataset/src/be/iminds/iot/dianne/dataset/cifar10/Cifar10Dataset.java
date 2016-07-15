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
package be.iminds.iot.dianne.dataset.cifar10;

import java.io.InputStream;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.dataset.FileDataset;

/**
 * The CIFAR-10 dataset, uses the binary images from:
 * http://www.cs.toronto.edu/~kriz/cifar.html
 * 
 * @author tverbele
 *
 */
@Component(
		service={Dataset.class},
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.CIFAR10",
		immediate=true)
public class Cifar10Dataset extends FileDataset {

	@Override
	protected void init(Map<String, Object> properties){
		this.name = "CIFAR-10";
		this.inputDims = new int[]{3, 32, 32};
		this.targetDims = new int[]{10};
		this.noSamples = 60000;
		
		this.labelsFile = "batches.meta.txt";
		this.inputFiles = new String[]{
				"data_batch_1.bin", "data_batch_2.bin", 
				"data_batch_3.bin", "data_batch_4.bin", 
				"data_batch_5.bin", "test_batch.bin"};
	}
	
	@Override
	protected void parse(InputStream in, InputStream targ) throws Exception {
		while(in.available()>0){
			int i = readUByte(in);
			targets[count][i] = 1;
			
			for(int j=0;j<inputSize;j++){
				inputs[count][j] = (float)readUByte(in)/255f;
			}
			
			count++;
		}
	}
	
}
