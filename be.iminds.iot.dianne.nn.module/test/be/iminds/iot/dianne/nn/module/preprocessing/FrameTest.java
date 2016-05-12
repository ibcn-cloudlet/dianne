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
 *     Tim Verbelen, Steven Bohez, Elias De Coninck
 *******************************************************************************/
package be.iminds.iot.dianne.nn.module.preprocessing;

import java.util.Arrays;
import java.util.UUID;

import org.junit.BeforeClass;
import org.junit.Test;

import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.tensor.NativeTensorLoader;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.util.ImageConverter;

public class FrameTest {

	private ImageConverter converter = new ImageConverter();

	@BeforeClass
	public static void load() {
		NativeTensorLoader loader = new NativeTensorLoader();
		loader.activate();
	}
	
	@Test
	public void testFrame() throws Exception {
	
		Frame frame = new Frame(3, 231, 231);
		
		Tensor input = converter.readFromFile("test/snake.jpg");
		converter.writeToFile("test/out.png", input);

		Object lock = new Object();
		frame.addForwardListener(new ForwardListener() {
			
			@Override
			public void onForward(UUID moduleId, Tensor output, String... tags) {
				System.out.println(Arrays.toString(output.dims()));
				try {
					converter.writeToFile("test/framed.png", output);
				} catch (Exception e) {
					e.printStackTrace();
				}
				synchronized(lock){
					lock.notifyAll();
				}
			}
		});
		long t1 = System.currentTimeMillis();
		frame.forward(UUID.randomUUID(), input);
		synchronized(lock){
			lock.wait();
		}
		long t2 = System.currentTimeMillis();
		System.out.println("Time "+(t2-t1)+" ms");
	}
}
