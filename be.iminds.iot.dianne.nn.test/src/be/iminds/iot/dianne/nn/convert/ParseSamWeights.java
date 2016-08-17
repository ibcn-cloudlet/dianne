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
package be.iminds.iot.dianne.nn.convert;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;

public class ParseSamWeights {

	public static void main(String[] args) throws Exception {

		String dir = "/home/tverbele/Desktop/sam/mnist";

		String[] linears = {
				"e5794329-b7e8-5925-5179-adddc98c963a",
				"2b445763-b022-5056-5df7-197ed068b887",
				"800bd472-2c32-5cb7-b7c0-ad1da59bac4f",
				"28961045-d6cd-82bf-bbe5-8093bad7f04f",
				"0d2789b3-34dd-6f5f-8b55-8dd5683fc81e"
				};
		String[] files = { "fc0", 
				"fc1", 
				//"fc2", 
				//"fc3",
				"0_softmax",
				"1_softmax",
				"2_softmax",
				//"3_softmax",
				//"4_softmax" 
				};
		String labels = "[0, 1, 2 , 3, 4, 5, 6, 7, 8, 9]";

	
		for (int i = 0; i < linears.length; i++) {
			// read weights
			String file = files[i];

			File w = new File(dir + file + "_W");
			
			String line;
			int count = 0;
			int noRows = 0;
			int noCols = 0;
			try (BufferedReader r = new BufferedReader(
					new InputStreamReader(new FileInputStream(w)))) {
				while ((line = r.readLine()) != null) {
					noRows++;
					if(noCols == 0)
						noCols = line.split(";").length;
				}
			}
			System.out.println(noRows+" "+noCols);
			float[][] weights = new float[noRows][noCols];
			
			int k=0;
			try (BufferedReader r = new BufferedReader(
					new InputStreamReader(new FileInputStream(w)))) {
				while ((line = r.readLine()) != null) {
					String[] split = line.split(";");
					for(int l=0;l<noCols;l++){
						weights[k][l] = Float.parseFloat(split[l]);
						count++;
					}
					k++;
				}
			}
			
			float[] bias = new float[noCols];
			File b = new File(dir + file + "_b");
			try (BufferedReader r = new BufferedReader(new InputStreamReader(
					new FileInputStream(b)))) {
				k = 0;
				while ((line = r.readLine()) != null) {
					bias[k] = Float.parseFloat(line);
					k++;
					count++;
				}
			}

			
			try (DataOutputStream out = new DataOutputStream(new FileOutputStream(new File(linears[i])))) {
				out.writeInt(noRows*noCols+noCols);
				for(int y=0;y<noCols;y++){
					for(int x=0;x<noRows;x++){
						out.writeFloat(weights[x][y]);
					}
				}
				for(int y=0;y<noCols;y++){
					out.writeFloat(bias[y]);
				}
			}
			
			System.out.println(count);
					
		}			

	}

}
