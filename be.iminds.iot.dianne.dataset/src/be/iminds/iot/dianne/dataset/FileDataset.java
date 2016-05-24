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


import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * This provides an abstract class to implement Datasets that read data 
 * from binary input files that contain raw data blobs. Specific Datasets should
 * implement the parse method to correctly parse out the input/output samples
 * into a float[][] array in memory.
 * 
 * @author tverbele
 *
 */
public abstract class FileDataset extends AbstractDataset {

	// load data in separate thread
	private ExecutorService loader = null;
	private Object lock = new Object();
	private int loaded = -1;
	protected int count = 0;

	
	// files with custom format
	protected String[] inputFiles;
	protected String[] outputFiles;
	
	protected float[][] inputs;
	protected float[][] outputs;
	
	protected boolean prefetch = false;
	
	@Override
	protected void activate(Map<String, Object> properties) {
		super.activate(properties);
		
		if(properties.containsKey("inputFiles")){
			inputFiles = (String[])properties.get("inputFiles");
		} else if(properties.containsKey("files")){
			inputFiles = (String[])properties.get("files");
		}
	
		if(properties.containsKey("outputFiles"))
			this.outputFiles = (String[])properties.get("outputFiles");

		String pf = (String)properties.get("prefetch");
		if(pf != null){
			prefetch = Boolean.parseBoolean(pf);
		}
		
		if(prefetch){
			load();
		}
	}
	
	private void load(){
		// allocate in float[][]
		inputs = new float[noSamples][inputSize];
		outputs = new float[noSamples][outputSize];
		
		// load all files on executor
		loader = Executors.newSingleThreadExecutor();
		loaded = 0;
		for(int i=0;i<inputFiles.length;i++){
			try {
				final InputStream ins = new BufferedInputStream(new FileInputStream(new File(dir+File.separator+inputFiles[i])));
				final InputStream outs;
				if(outputFiles != null){
					 outs = new BufferedInputStream(new FileInputStream(new File(dir+File.separator+outputFiles[i])));
				} else {
					outs = null;
				}
				final boolean last = i == inputFiles.length-1;
				loader.submit(() -> {
					try {
						parse(ins, outs);
						if(last){
							synchronized(lock){
								loaded = 1;
								loader.shutdown();
								loader = null;
								if(count != noSamples){
									System.err.println("Warning loading dataset "+name+": "+count+" samples loaded, "+noSamples+" expected");
									noSamples = count;
								}
								
								lock.notifyAll();
							}
						}
					} catch(Exception e){
						e.printStackTrace();
					}
				});
				
			} catch(Exception e){
				e.printStackTrace();
			}
		}
	}
	
	private void checkLoaded(){
		if(loaded < 1){
			if(loaded == -1)
				load();
			
			synchronized(lock){
				if(loaded != 1){
					try {
						lock.wait();
					} catch (InterruptedException e) {}
				}
			}
		}
	}
	
	@Override
	protected void readLabels(String file) {
		labels = new String[outputSize];
		try {
			InputStream labelInput = new FileInputStream(dir+File.separator+file);
				
			BufferedReader reader = new BufferedReader(new InputStreamReader(labelInput));
			for(int i=0;i<outputSize;i++){
				labels[i] = reader.readLine();
			}
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	protected abstract void parse(InputStream in, InputStream out) throws Exception;
				
	@Override
	public Tensor getInputSample(int index, Tensor t) {
		checkLoaded();

		if(t == null)
			t = new Tensor(inputs[index], inputDims);
		else 
			t.set(inputs[index]);
		return t;
	}

	@Override
	public Tensor getOutputSample(int index, Tensor t) {
		checkLoaded();

		if(t == null)
			t = new Tensor(outputs[index], outputDims);
		else 
			t.set(outputs[index]);
		return t;
	}
	
	protected int readInt(InputStream is) throws IOException{
		byte[] b = new byte[4];
		is.read(b, 0, 4);
		int i = ((0xFF & b[0]) << 24) | ((0xFF & b[1]) << 16) |
	            ((0xFF & b[2]) << 8) | (0xFF & b[3]);
		return i;
	}
	
	protected int readUByte(InputStream is) throws IOException{
		byte[] b = new byte[1];
		is.read(b, 0, 1);
		int i = (0xFF & b[0]);
		return i;
	}
}
