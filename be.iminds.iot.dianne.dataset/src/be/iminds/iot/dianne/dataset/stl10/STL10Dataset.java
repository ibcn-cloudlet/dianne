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
package be.iminds.iot.dianne.dataset.stl10;

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
		property={"name=STL-10","aiolos.unique=true"})
public class STL10Dataset extends GenericFileDataset {

	private int s = 0;

	@Override
	protected void init(Map<String, Object> properties){
		String d = (String)properties.get("be.iminds.iot.dianne.dataset.stl10.location");
		if(d!=null){
			this.dir = d;
		}

		this.name = "STL-10";
		this.inputDims = new int[]{3, 96, 96};
		this.outputDims = new int[]{10};
		this.noSamples = 13000;
		
		this.labelsFile = "class_names.txt";
		this.inputFiles = new String[]{
				"train_X.bin", "test_X.bin"};
		this.outputFiles = new String[]{
				"train_y.bin","test_y.bin"};
	}
	
	@Override
	protected void parse(InputStream in, InputStream out) throws Exception {
		while(in.available()>0
				&& out.available()>0){
			
			int i = readUByte(out);
			// categories are from 1..10
			outputs[s][i-1] = 1;

			
			// STL10 is formatted column-major, convert to row-major
			for(int c=0;c<3;c++){
				for(int y=0;y<96;y++){
					for(int x=0;x<96;x++){
						inputs[s][c*96*96+x*96+y] = (float)readUByte(in)/255f;
					}
				}
			}

			s++;
		}
	}
	
}
