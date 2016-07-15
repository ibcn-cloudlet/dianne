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


import java.io.InputStream;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;

/**
 * A generic FileDataset that treats input/target as a blob of unsigned bytes
 * 
 * @author tverbele
 *
 */
@Component(
		service={Dataset.class},
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.FileDataset",
		immediate=true, 
		property={"aiolos.unique=true"})
public class GenericFileDataset extends FileDataset {

	protected void init(Map<String, Object> properties){}
	
	protected void parse(InputStream in, InputStream targ) throws Exception{
		while(in.available()>0){
			if(targ != null){
				int i = readUByte(targ);
				targets[count][i] = 1;
			} else {
				int i = readUByte(in);
				targets[count][i] = 1;
			}
			
			for(int j=0;j<inputSize;j++){
				inputs[count][j] = (float)readUByte(in)/255f;
			}
			
			count++;
		}
	}
}