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
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

/**
 * The MNIST dataset, uses the images form LeCun's website:
 * http://yann.lecun.com/exdb/mnist/
 * 
 * @author tverbele
 *
 */
@Component(immediate=true, property={"name=MNIST","aiolos.unique=true"})
public class MNISTDataset implements Dataset{

	private TensorFactory factory;
	
	private List<Sample> data = new ArrayList<Sample>();
	private String[] labels = new String[]{"0","1","2","3","4","5","6","7","8","9"};
	
	private int noRows;
	private int noColumns;
	private int inputSize;
	private int outputSize;
	private int noSamples;
	
	private String dir = "";
	// thread to start loading data when constructed
	private ExecutorService loader = Executors.newSingleThreadExecutor();
	
	@Reference
	void setTensorFactory(TensorFactory f){
		this.factory = f;
	}
	
	@Activate
	public void activate(BundleContext context){
		String d = context.getProperty("be.iminds.iot.dianne.dataset.mnist.location");
		if(d!=null){
			this.dir = d;
		}

		// merge train and test samples into one dataset
		read("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
		read("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
	}
	
	private void read(String images, String labels){
		try {
			InputStream imageInput = new FileInputStream(dir+images);

			int magic = readInt(imageInput);
			assert magic == 2051;
			int noImages = readInt(imageInput);
			noRows = readInt(imageInput);
			noColumns = readInt(imageInput);
			
			InputStream labelInput = new FileInputStream(dir+labels);
			magic = readInt(labelInput);
			assert magic == 2049;
			int noLabels = readInt(labelInput);

			assert noLabels == noImages;
			
			noSamples += noImages;
			inputSize = noRows*noColumns;
			outputSize = 10;
			
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
	public Tensor getInputSample(int index) {
		// some hack to allow prefecting on construction
		// TODO better solution?
		while(data.size()<=index && index < noSamples){
			synchronized(data){
				if(data.size()<=index ){
					try {
						data.wait();
					} catch (InterruptedException e) {
					}
				}
			}
		}
		return data.get(index).input;
	}

	@Override
	public Tensor getOutputSample(int index) {
		// some hack to allow prefecting on construction
		// TODO better solution?
		while(data.size()<=index && index < noSamples){
			synchronized(data){
				if(data.size()<=index && index < noSamples){
					try {
						data.wait();
					} catch (InterruptedException e) {
					}
				}
			}
		}
		return data.get(index).output;
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
				//Tensor input = factory.createTensor(inputSize);
				Tensor output = factory.createTensor(outputSize);
				output.fill(0.0f);
				
				float inputData[] = new float[inputSize];
				for(int j=0;j<inputSize;j++){
					inputData[j] = (float)readUByte(imageInput)/255f;
				}
				Tensor input = factory.createTensor(inputData, noRows, noColumns);
				
				int i = readUByte(labelInput);
				output.set(1.0f, i);
				
				Sample s = new Sample(input, output);
				synchronized(data){
					data.add(s);
					data.notifyAll();
				}
			}
			
		} catch(Exception e){
			e.printStackTrace();
		}
		
	}
	
}
