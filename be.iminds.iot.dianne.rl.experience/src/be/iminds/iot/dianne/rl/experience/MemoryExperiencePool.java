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
import be.iminds.iot.dianne.api.dataset.Sequence;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolBatch;

@Component(
		service={ExperiencePool.class, Dataset.class},
		immediate=true, 
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.MemoryExperiencePool")
public class MemoryExperiencePool extends AbstractExperiencePool {

	private float[] samples;

	@Override
	protected void setup(Map<String, Object> config) {
		if(Integer.MAX_VALUE/sampleSize < maxSize){
			System.err.println("Failed to setup Experience Pool "+name+" in memory, maxSize "+maxSize+" too big to allocate in memory");
			throw new RuntimeException("Failed to instantiate experience pool, too big to allocate in memory");
		}
		try {
			samples = new float[maxSize*sampleSize];
		} catch(OutOfMemoryError e){
			System.err.println("Failed to setup Experience Pool "+name+" in memory: failed to allocate "+maxSize*sampleSize/1000000+" MB");
			throw new RuntimeException("Failed to instantiate experience pool, not enough memory");
		}
	}

	@Override
	protected void loadData(long position, float[] data) {
		System.arraycopy(samples, (int)position, data, 0, data.length);
	}

	@Override
	protected void writeData(long position, float[] data) {
		System.arraycopy(data, 0, samples, (int)position, data.length);
	}

	@Override
	protected void dumpData() throws IOException {
		try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(new File(dir+File.separator+"data.bin"))))){
			for(int i=0;i<samples.length;i++){
				out.writeFloat(samples[i]);
			}
			out.flush();
		}
	}

	@Override
	protected void recoverData() {
		try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(new File(dir+File.separator+"data.bin"))))){
			for(int i=0;i<samples.length;i++){
				samples[i] = in.readFloat();
			}
		} catch(Exception e){}
	}
}
