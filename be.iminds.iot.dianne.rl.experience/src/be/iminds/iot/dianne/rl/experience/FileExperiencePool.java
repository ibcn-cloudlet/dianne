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
		
		long bufferSize = ((long)maxSize)*((long)sampleSize)*4;
		
		int noBuffers = 1;
		if(bufferSize > SIZE_PER_BUFFER){
			noBuffers = (int) (bufferSize / SIZE_PER_BUFFER);
			if(bufferSize % SIZE_PER_BUFFER != 0)
				noBuffers++;
		}
		
		buffers = new FloatBuffer[noBuffers];

		try {
			for(int i=0;i<buffers.length;i++){
				buffers[i] = openFileAsFloatBuffer("buffer"+i+".bin", bufferSize > SIZE_PER_BUFFER ? SIZE_PER_BUFFER : bufferSize);
			}
		} catch(Exception e){
			e.printStackTrace();
		}
	}

	@Override
	protected void loadData(long position, float[] data) {
		int size = data.length;
		
		int b = (int)(position / FLOATS_PER_BUFFER);
		FloatBuffer buffer = buffers[b];
		int i = (int)(position % FLOATS_PER_BUFFER);

		if(FLOATS_PER_BUFFER-i < size){
			// split in two
			int half = FLOATS_PER_BUFFER-i;
			buffer.position(i);
			buffer.get(data, 0, half);
			
			FloatBuffer next = buffers[b+1];
			next.position(0);
			next.get(data, half, size-half);
		} else {
			buffer.position(i);
			buffer.get(data);
		}
	}

	@Override
	protected void writeData(long position, float[] data) {
		int b = (int) (position / FLOATS_PER_BUFFER);
		FloatBuffer buffer = buffers[b];
		int i = (int)(position % FLOATS_PER_BUFFER);

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

	
	@Override
	protected void dumpData() throws IOException {
		for(int i=0;i<buffers.length;i++){
			try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(new File(dir+File.separator+"data"+i+".bin"))))){
				for(int k=0;k<buffers[i].limit();k++){
					buffers[i].position(k);
					out.writeFloat(buffers[i].get());
				}
				out.flush();
			}
		}
	}

	@Override
	protected void recoverData() {
		for(int i=0;i<buffers.length;i++){
			try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(new File(dir+File.separator+"data"+i+".bin"))))){
				int k=0;
				while(true){
					buffers[i].position(k++);
					buffers[i].put(in.readFloat());
				}
			} catch(Exception e){}
		}
	}
}
