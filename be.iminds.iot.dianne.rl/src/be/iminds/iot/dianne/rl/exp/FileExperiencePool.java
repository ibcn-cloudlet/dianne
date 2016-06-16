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
package be.iminds.iot.dianne.rl.exp;

import java.io.File;
import java.io.RandomAccessFile;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(
		service={ExperiencePool.class, Dataset.class},
		immediate=true, 
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.FileExperiencePool")
public class FileExperiencePool extends AbstractExperiencePool {

	private FloatBuffer[] buffers;
	private int samplesPerBuffer;
	
	private ReentrantReadWriteLock lock = new ReentrantReadWriteLock(true);
	
	@Override
	protected void setup() {
		long bufferSize = sampleSize;
		bufferSize *= maxSize;
		bufferSize *= 4;
		
		int noBuffers = 1;
		samplesPerBuffer = (int)bufferSize;
		if(bufferSize > Integer.MAX_VALUE){
			noBuffers = (int) (bufferSize / Integer.MAX_VALUE) + 1;
			samplesPerBuffer = Integer.MAX_VALUE/(4*sampleSize);
		}
		
		buffers = new FloatBuffer[noBuffers];

		try {
			for(int i=0;i<buffers.length;i++){
				buffers[i] = openFileAsFloatBuffer("data"+i+".bin", bufferSize > Integer.MAX_VALUE ? Integer.MAX_VALUE : bufferSize);
			}
		} catch(Exception e){
			e.printStackTrace();
		}
	}

	@Override
	protected float[] loadData(int index) {
		float[] data = new float[sampleSize];
		FloatBuffer buffer = buffers[index / samplesPerBuffer];
		int i = index % samplesPerBuffer;
		try {
			lock.readLock().lock();
			int position = i*data.length;
			buffer.position(position);
			buffer.get(data);
		} catch(Exception e){
			e.printStackTrace();
			System.exit(-1);
		} finally {
			lock.readLock().unlock();
		}
		return data;
	}

	@Override
	protected void writeData(int index, float[] data) {
		FloatBuffer buffer = buffers[index / samplesPerBuffer];
		int i = index % samplesPerBuffer;
		try {
			lock.writeLock().lock();
			int position = i*data.length;
			buffer.position(position);
			buffer.put(data);
		} catch(Exception e){
			e.printStackTrace();
			System.exit(-1);
		} finally {
			lock.writeLock().unlock();
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
