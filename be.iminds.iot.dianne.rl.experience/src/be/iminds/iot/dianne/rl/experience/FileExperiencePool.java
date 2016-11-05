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

import java.io.File;
import java.io.RandomAccessFile;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;

@Component(
		service={ExperiencePool.class, Dataset.class},
		immediate=true, 
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.FileExperiencePool")
public class FileExperiencePool extends AbstractExperiencePool {

	private FloatBuffer[] buffers;
	private int SIZE_PER_BUFFER = Integer.MAX_VALUE;
	private int FLOATS_PER_BUFFER = Integer.MAX_VALUE/4;
	
	@Override
	protected void setup(Map<String, Object> config) {
		if(config.containsKey("sizePerBuffer")){
			SIZE_PER_BUFFER = Integer.parseInt(config.get("sizePerBuffer").toString());
		}
		
		FLOATS_PER_BUFFER = SIZE_PER_BUFFER/4;
		
		long bufferSize = maxSize;
		
		int noBuffers = 1;
		if(bufferSize > SIZE_PER_BUFFER){
			noBuffers = (int) (bufferSize / SIZE_PER_BUFFER);
			if(bufferSize % SIZE_PER_BUFFER != 0)
				noBuffers++;
		}
		
		buffers = new FloatBuffer[noBuffers];

		try {
			for(int i=0;i<buffers.length;i++){
				buffers[i] = openFileAsFloatBuffer("data"+i+".bin", bufferSize > SIZE_PER_BUFFER ? SIZE_PER_BUFFER : bufferSize);
			}
		} catch(Exception e){
			e.printStackTrace();
		}
	}

	@Override
	protected float[] loadData(int index) {
		float[] data = new float[sampleSize];
				
		int b = index / FLOATS_PER_BUFFER;
		FloatBuffer buffer = buffers[b];
		int i = index % FLOATS_PER_BUFFER;

		if(FLOATS_PER_BUFFER-i < sampleSize){
			// split in two
			int half = FLOATS_PER_BUFFER-i;
			buffer.position(i);
			buffer.get(data, 0, half);
			
			FloatBuffer next = buffers[b+1];
			next.position(0);
			next.get(data, half, sampleSize-half);
		} else {
			buffer.position(i);
			buffer.get(data);
		}
		
		return data;
	}

	@Override
	protected void writeData(int index, float[] data) {
		int b = index / FLOATS_PER_BUFFER;
		FloatBuffer buffer = buffers[b];
		int i = index % FLOATS_PER_BUFFER;

		if(FLOATS_PER_BUFFER-i < data.length){
			// split in two
			int half = FLOATS_PER_BUFFER-i;
			buffer.position(i);
			buffer.put(data, 0, half);
			
			FloatBuffer next = buffers[b+1];
			next.position(0);
			next.put(data, half, data.length-half);
		} else {
			buffer.position(i);
			buffer.put(data);
		}
		
	}
	
	private FloatBuffer openFileAsFloatBuffer(String fileName, long size) throws Exception {
		File file = new File(dir+File.separator+fileName);
		file.delete();
		try (FileChannel fc = new RandomAccessFile(file, "rw").getChannel()){
			MappedByteBuffer buffer = fc.map(FileChannel.MapMode.READ_WRITE, 0, size);
			buffer.order(ByteOrder.nativeOrder());
			buffer.load();
			FloatBuffer floatBuffer = buffer.asFloatBuffer();
			return floatBuffer;
		} catch(Exception e){
			e.printStackTrace();
			throw e;
		}
	}

}
