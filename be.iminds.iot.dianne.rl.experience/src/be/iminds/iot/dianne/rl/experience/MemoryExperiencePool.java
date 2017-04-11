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
package be.iminds.iot.dianne.rl.experience;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;

@Component(
		service={ExperiencePool.class, Dataset.class},
		immediate=true, 
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.MemoryExperiencePool")
public class MemoryExperiencePool extends AbstractExperiencePool {

	private final static int MAX_BUFFER_SIZE = 1000000000;
	
	private float[][] buffers;
	private int bufferSize;
	
	@Override
	protected void setup(Map<String, Object> config) {
		int maxSamplesPerBuffer = MAX_BUFFER_SIZE/sampleSize;
		int noBuffers = maxSize  / maxSamplesPerBuffer;
		if(maxSize % maxSamplesPerBuffer != 0)
			noBuffers += 1;
		
		bufferSize = noBuffers == 1 ? maxSize*sampleSize : maxSamplesPerBuffer*sampleSize;
		
		try {
			buffers = new float[noBuffers][bufferSize];
		} catch(OutOfMemoryError e){
			System.err.println("Failed to setup Experience Pool "+name+" in memory: failed to allocate "+(bufferSize/1000000)*noBuffers+" MB");
			throw new RuntimeException("Failed to instantiate experience pool, not enough memory", e);
		}
	}

	@Override
	protected void loadData(long position, float[] data) {
		int buffer = (int)(position / bufferSize);
		int pos = (int)(position % bufferSize);
		
		if(pos+data.length <= bufferSize){
			System.arraycopy(buffers[buffer], pos, data, 0, data.length);
		} else {
			int s = bufferSize-pos;
			System.arraycopy(buffers[buffer], pos, data, 0, s);
			System.arraycopy(buffers[buffer+1], 0, data, s+1, data.length-s);
		}
	}

	@Override
	protected void writeData(long position, float[] data) {
		int buffer = (int)(position / bufferSize);
		int pos = (int)(position % bufferSize);
		
		if(pos+data.length <= bufferSize){
			System.arraycopy(data, 0, buffers[buffer], pos, data.length);
		} else {
			int s = bufferSize-pos;
			System.arraycopy(data, 0, buffers[buffer], pos, s);
			System.arraycopy(data, s+1, buffers[buffer+1], 0, data.length-s);
		}
	}

	@Override
	protected void dumpData() throws IOException {
		for(int k=0;k<buffers.length;k++){
			float[] buffer = buffers[k];
			try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(new File(dir+File.separator+"data-"+k+".bin"))))){
				for(int i=0;i<buffer.length;i++){
					out.writeFloat(buffer[i]);
				}
				out.flush();
			}
		}
	}

	@Override
	protected void recoverData() {
		for(int k=0;k<buffers.length;k++){
			float[] buffer = buffers[k];
			try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(new File(dir+File.separator+"data-"+k+".bin"))))){
				for(int i=0;i<buffer.length;i++){
					buffer[i] = in.readFloat();
				}
			} catch(Exception e){}
		}
	}
}
