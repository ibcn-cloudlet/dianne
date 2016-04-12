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
package be.iminds.iot.dianne.tensor.util;

import org.junit.Test;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.dianne.tensor.impl.java.JavaTensorFactory;

public class ImageConverterTest {
	
	private TensorFactory factory = new JavaTensorFactory();

	@Test
	public void testImageNetImages() {
		ImageConverter conv = new ImageConverter(factory);
		long t1 = System.currentTimeMillis();
		int start = 0;
		int n = 100;
		for(int i=start;i<start+n;i++){
			String dir = "../tools/datasets/ImageNet/";
			String file = dir + "images/" + "ILSVRC2012_val_"
					+ String.format("%08d", i+1) + ".JPEG";
			try {
				conv.readFromFile(file);
			} catch(Exception e){
				System.out.println("Error with image "+file);
				e.printStackTrace();
			}
		}
		long t2 = System.currentTimeMillis();
		System.out.println("Avg read time: "+(t2-t1)/n+" ms");
	}
	
	@Test
	public void testReadWriteImage() throws Exception {
		ImageConverter conv = new ImageConverter(factory);

		int i = 0;
		String dir = "../tools/datasets/ImageNet/";
		String file = dir + "images/" + "ILSVRC2012_val_"
				+ String.format("%08d", i+1) + ".JPEG";
	
		Tensor t = conv.readFromFile(file);
		conv.writeToFile("test.jpg", t);
	}
}
