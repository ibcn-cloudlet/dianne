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
@Component(immediate=true, property={"name=CIFAR-10","aiolos.unique=true"})
public class Cifar10Dataset implements Dataset {

	private float[][] inputs;
	private float[][] outputs;
	private int loaded = 0;
	
	private String[] labels;
	
	private int[] inputDims = new int[]{3, 32, 32};
	private int inputSize = 3*32*32;
	private int[] outputDims = new int[]{10};
	private int outputSize = 10;
	private int noSamples = 60000;
	
	private String dir = "";
	// thread to start loading data when constructed
	private ExecutorService loader = Executors.newSingleThreadExecutor();
	
	@Activate
	public void activate(BundleContext context){
		String d = context.getProperty("be.iminds.iot.dianne.dataset.cifar10.location");
		if(d!=null){
			this.dir = d;
		}
		
		inputs = new float[noSamples][inputSize];
		outputs = new float[noSamples][outputSize];
		
		readLabels("batches.meta.txt");
		// merge all samples into one dataset
		read("data_batch_1.bin");
		read("data_batch_2.bin");
		read("data_batch_3.bin");
		read("data_batch_4.bin");
		read("data_batch_5.bin");
		read("test_batch.bin");
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
			InputStream input = new FileInputStream(dir+file);
			
			loader.execute(new Runnable() {
				@Override
				public void run() {
					parse(input);
				}
			});
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	@Override
	public String getName(){
		return "CIFAR-10";
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
	
	private void parse(InputStream input) {
		try {
			while(input.available()>0){
				
				int i = readUByte(input);
				outputs[loaded][i] = 1;
				
				for(int j=0;j<inputSize;j++){
					inputs[loaded][j] = (float)readUByte(input)/255f;
				}
				
				loaded++;
			}
			
		} catch(Exception e){
			e.printStackTrace();
		}
		
	}
	
}
