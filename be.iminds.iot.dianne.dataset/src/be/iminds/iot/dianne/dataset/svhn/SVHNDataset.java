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
package be.iminds.iot.dianne.dataset.svhn;

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
 * The Street View House Numbers dataset, based on the cropped 32x32 images
 * http://ufldl.stanford.edu/housenumbers/
 * 
 * @author tverbele
 *
 */
@Component(immediate=true, property={"name=SVHN","aiolos.unique=true"})
public class SVHNDataset implements Dataset{

	private float[][] inputs;
	private float[][] outputs;
	private int loaded = 0;
	
	private String[] labels = new String[]{"0","1","2","3","4","5","6","7","8","9"};
	
	private int[] inputDims = new int[]{3, 32, 32};
	private int inputSize = 3*32*32;
	private int[] outputDims = new int[]{10};
	private int outputSize = 10;
	private int noSamples = 73257+26032;
	
	
	private String dir = "";
	// thread to start loading data when constructed
	private ExecutorService loader = Executors.newSingleThreadExecutor();
	
	@Activate
	public void activate(BundleContext context){
		String d = context.getProperty("be.iminds.iot.dianne.dataset.svhn.location");
		if(d!=null){
			this.dir = d;
		}
		
		inputs = new float[noSamples][inputSize];
		outputs = new float[noSamples][outputSize];

		// merge train and test samples into one dataset
		read("train_images.bin", "train_labels.bin", 73257);
		read("test_images.bin", "test_labels.bin", 26032);
	}
	
	@Override
	public int[] inputDims(){
		return inputDims;
	}
	
	@Override
	public int[] outputDims(){
		return outputDims;
	}
	
	private void read(String images, String labels, int count){
		try {
			InputStream imageInput = new FileInputStream(dir+images);
			InputStream labelInput = new FileInputStream(dir+labels);
			
			loader.execute(new Runnable() {
				@Override
				public void run() {
					parse(imageInput, labelInput, count);
				}
			});
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	@Override
	public String getName(){
		return "SVHN";
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
	
	private int readUByte(InputStream is) throws IOException{
		byte[] b = new byte[1];
		is.read(b, 0, 1);
		int i = (0xFF & b[0]);
		return i;
	}
	
	private void parse(InputStream imageInput, InputStream labelInput, int count) {
		try {
			for(int k=0;k<count;k++){
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
