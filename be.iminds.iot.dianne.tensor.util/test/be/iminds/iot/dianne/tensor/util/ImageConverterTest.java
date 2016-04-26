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

	/**
	 * @throws Exception
	 */
	@Test
	public void testImageNetImages() throws Exception {
		ImageConverter conv = new ImageConverter(factory);
		long t1 = System.currentTimeMillis();
		int n = 100;
		for(int i=0;i<n;i++){
			String file = String.format("../tools/datasets/ImageNet/images/ILSVRC2012_val_%08d.JPEG", 1);
			conv.readFromFile(file);
		}
		long t2 = System.currentTimeMillis();
		System.out.println("Avg read time: "+(t2-t1)/n+" ms");
	}
	
	/**
	 * @throws Exception
	 */
	@Test
	public void testReadWriteImage() throws Exception {
		ImageConverter conv = new ImageConverter(factory);
		Tensor t = conv.readFromFile(String.format("../tools/datasets/ImageNet/images/ILSVRC2012_val_%08d.JPEG", 1));
		conv.writeToFile("generated/test.jpg", t);
	}
}
