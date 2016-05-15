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

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * The CIFAR-10 dataset, uses the binary images from:
 * http://www.cs.toronto.edu/~kriz/cifar.html
 * 
 * @author tverbele
 *
 */
@Component(immediate=true, property={"name=STL-10","aiolos.unique=true"})
public class STL10Dataset implements Dataset {

	private float[][] inputs;
	private float[][] outputs;
	private int loaded = 0;
	
	private String[] labels;
	
	private int[] inputDims = new int[]{3, 96, 96};
	private int inputSize = 3*96*96;
	private int[] outputDims = new int[]{10};
	private int outputSize = 10;
	private int noSamples = 13000;
	
	private String dir = "";
	// thread to start loading data when constructed
	private ExecutorService loader = Executors.newSingleThreadExecutor();
	
	@Activate
	public void activate(BundleContext context){
		String d = context.getProperty("be.iminds.iot.dianne.dataset.stl10.location");
		if(d!=null){
			this.dir = d;
		}
		
		inputs = new float[noSamples][inputSize];
		outputs = new float[noSamples][outputSize];
		
		readLabels("class_names.txt");
		// merge all samples into one dataset
		read("train");
		read("test");
	}
	
	@Override
	public int[] inputDims(){
		return inputDims;
	}
	
	@Override
	public int[] outputDims(){
		return outputDims;
	}
	
	private void readLabels(String file){
		try {
			InputStream labelInput = new FileInputStream(dir+file);
			
			labels = new String[10];
			BufferedReader reader = new BufferedReader(new InputStreamReader(labelInput));
			for(int i=0;i<10;i++){
				labels[i] = reader.readLine();
			}
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	private void read(String file){
		try {
			InputStream imageInput = new FileInputStream(dir+file+"_X.bin");
			InputStream labelInput = new FileInputStream(dir+file+"_y.bin");
			
			loader.execute(new Runnable() {
				@Override
				public void run() {
					parse(imageInput, labelInput);
				}
			});
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	@Override
	public String getName(){
		return "STL-10";
	}
	
	@Override
	public int size() {
		return noSamples;
	}

	@Override
	public Tensor getInputSample(int index, Tensor t) {
		// some hack to allow prefecting on construction
		// TODO better solution?
		while(loaded<index && index < noSamples){
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {}
		}

		if(t == null)
			t = new Tensor(inputs[index], inputDims);
		else 
			t.set(inputs[index]);
		return t;
	}

	@Override
	public Tensor getOutputSample(int index, Tensor t) {
		// some hack to allow prefecting on construction
		// TODO better solution?
		while(loaded<index && index < noSamples){
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {}
		}

		if(t == null)
			t = new Tensor(outputs[index], outputDims);
		else 
			t.set(outputs[index]);
		return t;
	}
	
	@Override
	public String[] getLabels(){
		return labels;
	}

	private int readInt(InputStream is) throws IOException{
		byte[] b = new byte[4];
		is.read(b, 0, 4);
		int i = ((0xFF & b[0]) << 24) | ((0xFF & b[1]) << 16) |
	            ((0xFF & b[2]) << 8) | (0xFF & b[3]);
		return i;
	}
	
	private int readUByte(InputStream is) throws IOException{
		byte[] b = new byte[1];
		is.read(b, 0, 1);
		int i = (0xFF & b[0]);
		return i;
	}
	
	private void parse(InputStream imageInput, InputStream labelInput) {
		try {
			while(imageInput.available()>0
					&& labelInput.available()>0){
				
				int i = readUByte(labelInput);
				// categories are from 1..10
				outputs[loaded][i-1] = 1;

				
				// STL10 is formatted column-major, convert to row-major
				for(int c=0;c<3;c++){
					for(int y=0;y<96;y++){
						for(int x=0;x<96;x++){
							inputs[loaded][c*96*96+x*96+y] = (float)readUByte(imageInput)/255f;
						}
					}
				}

				loaded++;
			}
		} catch(Exception e){
			e.printStackTrace();
		}
		
	}
	
}
