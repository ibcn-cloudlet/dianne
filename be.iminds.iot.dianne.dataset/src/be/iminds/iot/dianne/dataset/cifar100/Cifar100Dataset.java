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
package be.iminds.iot.dianne.dataset.cifar100;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
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
 * The CIFAR-100 dataset, uses the binary images from:
 * http://www.cs.toronto.edu/~kriz/cifar.html
 * 
 * @author tverbele
 *
 */
@Component(immediate=true, property={"name=CIFAR-100","aiolos.callback=be.iminds.iot.dianne.api.dataset.Dataset"})
public class Cifar100Dataset implements Dataset {

	private TensorFactory factory;
	
	private List<Sample> data = new ArrayList<Sample>();
	private String[] labels;
	
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
		String d = context.getProperty("be.iminds.iot.dianne.dataset.cifar100.location");
		if(d!=null){
			this.dir = d;
		}
		// TODO should be able to have the two at the same time?
		String l = context.getProperty("be.iminds.iot.dianne.dataset.cifar100.labels");
		if(l!=null){
			if(l.equals("coarse")){
				outputSize = 20;
			} else if(l.equals("fine")){
				outputSize = 100;
			}
		} else {
			outputSize = 100;
		}
		
		noRows = 32;
		noColumns = 32;
		inputSize = noRows*noColumns*3;
		noSamples = 60000;
		
		if(outputSize==100){
			readLabels("fine_label_names.txt");
		} else {
			readLabels("coarse_label_names.txt");
		}

		// merge all samples into one dataset
		read("test.bin");
		read("train.bin");
	}
	
	private void readLabels(String file){
		try {
			InputStream labelInput = new FileInputStream(dir+file);
			
			ArrayList<String> l = new ArrayList<String>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(labelInput));
			String s;
			while((s = reader.readLine()) != null ){
				l.add(s);
			}
			labels = new String[l.size()];
			for(int i=0;i<l.size();i++){
				labels[i] = l.get(i);
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
		return "CIFAR-100";
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
	
	private void parse(InputStream input) {
		try {
			while(input.available()>0){
				Tensor out = factory.createTensor(outputSize);
				out.fill(0.0f);
				
				int i1 = readUByte(input);
				int i2 = readUByte(input);
				if(outputSize==100){
					out.set(1.0f, i2);
				} else {
					out.set(1.0f, i1);
				}
				
				float inputData[] = new float[inputSize];
				for(int j=0;j<inputSize;j++){
					inputData[j] = (float)readUByte(input)/255f;
				}
				Tensor in = factory.createTensor(inputData, 3, noRows, noColumns);
				
				Sample s = new Sample(in, out);
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
