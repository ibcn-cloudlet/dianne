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
package be.iminds.iot.dianne.nn.test.onnx;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.osgi.framework.ServiceReference;

import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.nn.test.DianneTest;
import be.iminds.iot.dianne.onnx.api.OnnxConverter;
import be.iminds.iot.dianne.tensor.Tensor;
import junit.framework.Assert;


public class OnnxTest extends DianneTest {
	
	private OnnxConverter converter;
	
	@Override
	public void setUp() throws Exception {
    	super.setUp();
    	
       	ServiceReference r =  context.getServiceReference(OnnxConverter.class.getName());
    	converter = (OnnxConverter) context.getService(r);
    }
	
	public void testSingleFF() throws Exception {
		runTest("singleff");
	}

	public void testSingleConv() throws Exception {
		runTest("singleconv");
	}
	
	public void testConvLinear() throws Exception {
		runTest("convlinear");
	}
	
	public void testConvLinearPool() throws Exception {
		runTest("convlinearpool");
	}
	
	private void runTest(String test) throws Exception {
		NeuralNetworkDTO dto = converter.fromOnnx("onnxtest"+File.separator+test+File.separator+"model.pb");
		NeuralNetwork nn = deployNN(dto.name);
		Tensor input = toTensor("onnxtest"+File.separator+test+File.separator+"input.csv");
		Tensor expected = toTensor("onnxtest"+File.separator+test+File.separator+"output.csv");
		Tensor result = nn.forward(input);
		result.reshape(expected.dims());
		Assert.assertEquals(expected, result);	
	}
	
	private Tensor toTensor(String csv) {
		try(BufferedReader reader = new BufferedReader(
				new InputStreamReader(new FileInputStream(new File(csv))))){
			List<Float> data = new ArrayList<>();
			int width = 0;
			int height = 0;
			
			String line;
			while ((line = reader.readLine()) != null) {				
				String[] f = line.split(",");
				for(int j=0;j<f.length;j++) {
					data.add(Float.parseFloat(f[j]));
				}
				width = f.length;
				height++;
			}
			
			float[] array = new float[data.size()];
			for(int i=0;i<data.size();i++) {
				array[i] = data.get(i);
			}
			Tensor t = new Tensor(array, height, width);
			return t;
		} catch(Exception e) {
			throw new RuntimeException("Failed to read tensor "+csv);
		}
	}
}
