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

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.GenericFileDataset;

/**
 * The CIFAR-10 dataset, uses the binary images from:
 * http://www.cs.toronto.edu/~kriz/cifar.html
 * 
 * @author tverbele
 *
 */
@Component(
		service={Dataset.class},
		immediate=true, 
		property={"name=CIFAR-10","aiolos.unique=true"})
public class Cifar10Dataset extends GenericFileDataset {

	private int s = 0;

	@Override
	protected void init(Map<String, Object> properties){
		String d = (String)properties.get("be.iminds.iot.dianne.dataset.cifar10.location");
		if(d!=null){
			this.dir = d;
		}

		this.name = "CIFAR-10";
		this.inputDims = new int[]{3, 32, 32};
		this.outputDims = new int[]{10};
		this.noSamples = 60000;
		
		this.labelsFile = "batches.meta.txt";
		this.inputFiles = new String[]{
				"data_batch_1.bin", "data_batch_2.bin", 
				"data_batch_3.bin", "data_batch_4.bin", 
				"data_batch_5.bin", "test_batch.bin"};
	}
	
	@Override
	protected void parse(InputStream in, InputStream out) throws Exception {
		while(in.available()>0){
			int i = readUByte(in);
			outputs[s][i] = 1;
			
			for(int j=0;j<inputSize;j++){
				inputs[s][j] = (float)readUByte(in)/255f;
			}
			
			s++;
		}
	}
	
}
