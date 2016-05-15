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

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * The MNIST dataset, uses the images form LeCun's website:
 * http://yann.lecun.com/exdb/mnist/
 * 
 * @author tverbele
 *
 */
@Component(immediate=true, property={"name=MNIST","aiolos.unique=true"})
public class MNISTDataset implements Dataset{

	private float[][] inputs;
	private float[][] outputs;
	private int loaded = 0;
	
	private String[] labels = new String[]{"0","1","2","3","4","5","6","7","8","9"};
	
	private int[] inputDims = new int[]{1, 28, 28};
	private int inputSize = 28*28;
	private int[] outputDims = new int[]{10};
	private int outputSize = 10;
	private int noSamples = 70000;
	
	
	private String dir = "";
	// thread to start loading data when constructed
	private ExecutorService loader = Executors.newSingleThreadExecutor();
	
	@Activate
	public void activate(BundleContext context){
		String d = context.getProperty("be.iminds.iot.dianne.dataset.mnist.location");
		if(d!=null){
			this.dir = d;
		}
		
		inputs = new float[noSamples][inputSize];
		outputs = new float[noSamples][outputSize];

		// merge train and test samples into one dataset
		read("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
		read("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
	}
	
	@Override
	public int[] inputDims(){
		return inputDims;
	}
	
	@Override
	public int[] outputDims(){
		return outputDims;
	}
	
	private void read(String images, String labels){
		try {
			InputStream imageInput = new FileInputStream(dir+images);

			int magic = readInt(imageInput);
			assert magic == 2051;
			int noImages = readInt(imageInput);
			int noRows = readInt(imageInput);
			int noColumns = readInt(imageInput);
			
			InputStream labelInput = new FileInputStream(dir+labels);
			magic = readInt(labelInput);
			assert magic == 2049;
			int noLabels = readInt(labelInput);

			assert noLabels == noImages;
			
			loader.execute(new Runnable() {
				@Override
				public void run() {
					parse(imageInput, labelInput, noImages);
				}
			});
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	@Override
	public String getName(){
		return "MNIST";
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
	
	private void parse(InputStream imageInput, InputStream labelInput, int count) {
		try {
			for(int read = 0;read<count;read++){
				
				for(int j=0;j<inputSize;j++){
					inputs[loaded][j] = (float)readUByte(imageInput)/255f;
				}
				
				int i = readUByte(labelInput);
				outputs[loaded][i] = 1;
				
				loaded++;
			}
			
		} catch(Exception e){
			e.printStackTrace();
		}
		
	}
	
}
