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
package be.iminds.iot.dianne.api.dataset;


import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * This provides a generic abstract class to implement Datasets that read data 
 * from binary input files that contain raw data blobs. Specific Datasets should
 * implement the parse method to correctly parse out the input/output samples
 * into a float[][] array in memory.
 * 
 * @author tverbele
 *
 */
public abstract class GenericFileDataset implements Dataset {

	// load data in separate thread
	private ExecutorService loader = null;
	private Object lock = new Object();
	private int loaded = -1;
	
	protected String name;
	protected int[] inputDims;
	protected int inputSize;
	protected int[] outputDims;
	protected int outputSize;
	protected int noSamples;
	protected String[] labels;
	
	// files with custom format
	protected String[] inputFiles;
	protected String[] outputFiles;
	
	protected float[][] inputs;
	protected float[][] outputs;
	
	protected boolean prefetch = false;
	
	protected String dir = "";
	
	@Activate
	protected void activate(Map<String, Object> properties, BundleContext context) {
		String d = context.getProperty("be.iminds.iot.dianne.datasets.location");
		if(d != null){
			dir = d;
		}
		
		this.name = (String)properties.get("name");

		String[] id = (String[])properties.get("inputDims");
		if(id!=null){
			inputDims= new int[id.length];
			for(int i=0;i<id.length;i++){
				inputDims[i] = Integer.parseInt(id[i]);
			}
		}
		
		String[] od = (String[])properties.get("outputDims");
		if(od != null){
			outputDims= new int[od.length];
			for(int i=0;i<od.length;i++){
				outputDims[i] = Integer.parseInt(od[i]);
			}
		}
		
		String ns = (String)properties.get("noSamples");
		if(ns != null)
			noSamples = Integer.parseInt(ns);
		
		labels = (String[])properties.get("labels");
		
		inputFiles = (String[])properties.get("inputFiles");
		if(inputFiles == null){
			inputFiles = (String[])properties.get("files");
		}
		outputFiles = (String[])properties.get("outputFiles");

		String pf = (String)properties.get("prefetch");
		if(pf != null){
			prefetch = Boolean.parseBoolean(pf);
		}
		
		init(new ComboMap(properties, context));
		
		inputSize = 1;
		for(int i=0;i<inputDims.length;i++){
			inputSize *= inputDims[i];
		}
		
		outputSize = 1;
		for(int i=0;i<outputDims.length;i++){
			outputSize *= outputDims[i];
		}
		
		// allocate in float[][]
		inputs = new float[noSamples][inputSize];
		outputs = new float[noSamples][outputSize];
		
		if(prefetch){
			load();
		}
	}
	
	private void load(){
		// load all files on executor
		loader = Executors.newSingleThreadExecutor();
		loaded = 0;
		for(int i=0;i<inputFiles.length;i++){
			try {
				final InputStream ins = new BufferedInputStream(new FileInputStream(new File(inputFiles[i])));
				final InputStream outs;
				if(outputFiles != null){
					 outs = new BufferedInputStream(new FileInputStream(new File(outputFiles[i])));
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
	
	protected abstract void init(Map<String, Object> properties);
	
	protected abstract void parse(InputStream inputs, InputStream outputs) throws Exception;			
				
	@Override
	public String getName(){
		return name;
	}
	
	@Override
	public int[] inputDims(){
		return inputDims;
	}
	
	@Override
	public int[] outputDims(){
		return outputDims;
	}
	
	@Override
	public int size() {
		return noSamples;
	}

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
	
	@Override
	public String[] getLabels(){
		return labels;
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
	
	private class ComboMap extends HashMap<String, Object> {
		
		private final Map<String, Object> componentProperties;
		private final BundleContext bundleContext;
		
		public ComboMap(Map<String, Object> properties, BundleContext context){
			super();
			this.componentProperties = properties;
			this.bundleContext = context;
		}
		
		@Override
		public Object get(Object key){
			Object o = componentProperties.get(key);
			if(o != null)
				return o;
			
			return bundleContext.getProperty((String)key);
		}
	}
}
