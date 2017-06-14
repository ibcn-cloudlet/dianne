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
package be.iminds.iot.dianne.dataset.csv;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.dataset.FileDataset;

/**
 * Generic CSV Dataset
 * 
 * @author tverbele
 *
 */
@Component(
		service={Dataset.class},
		immediate=true, 
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.CSVDataset")
public class CSVDataset extends FileDataset{

	private String separator = ",";
	private int inputOffset = 0;
	private int targetOffset = 0;
	private boolean classification = false;
	
	@Override
	protected void init(Map<String, Object> properties){
		String file = (String)properties.get("file");
		if(file != null)
			inputFiles = new String[]{file};
		
		String s = (String)properties.get("separator");
		if(s != null){
			separator = s;
		}
		
		// how many columns to skip before input begins
		String io = (String)properties.get("inputOffset");
		if(io != null){
			inputOffset = Integer.parseInt(io.trim());
		}
		
		// how many columns to skip before target begins
		String to = (String)properties.get("targetOffset");
		if(to != null){
			targetOffset = Integer.parseInt(to.trim());
		}
		
		// check whether it is a classification problem or not
		String c = (String)properties.get("classification");
		if(c != null){
			classification = Boolean.parseBoolean(c);
		}
		
		if(!properties.containsKey("noSamples")){
			// count line numbers in file
			try {
				File f = new File(dir+File.separator+file);
				LineNumberReader lineNumberReader = new LineNumberReader(new FileReader(f));
				lineNumberReader.skip(Long.MAX_VALUE);
				noSamples = lineNumberReader.getLineNumber();
				lineNumberReader.close();
			} catch(Exception e){ }
		}
	}
	
	@Override
	protected void parse(InputStream in, InputStream targ) throws Exception{
		BufferedReader reader = new BufferedReader(new InputStreamReader(in));
		String s;
		while ((s = reader.readLine()) != null) {
			String[] data = s.split(separator);
			int i=inputOffset;
			for(;i<inputOffset+inputSize;i++){
				inputs[count][i] = Float.parseFloat(data[i]);
			}
			i += targetOffset;
			if(classification){
				// threat as class index?
				int index = Integer.parseInt(data[i]);
				targets[count][index] = 1;
			} else {
				for(int k=0;k<targetSize;k++){
					targets[count][k] = Float.parseFloat(data[i+k]);
				}
			}
			count++;
		}
	}
}
