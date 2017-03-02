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
package be.iminds.iot.dianne.tensor;

import java.util.ArrayList;
import java.util.List;

import org.junit.BeforeClass;
import org.junit.Test;

// small test for checking whether GC callbacks work...
public class MemoryTest {

	@BeforeClass
	public static void load() {
		NativeTensorLoader loader = new NativeTensorLoader();
		loader.activate(null);
	}
	
	@Test
	public void testGC() throws InterruptedException{
		List<Tensor> keep = new ArrayList<>();
		Thread t = new Thread(()->{
			
			for(int i=0;i<1024;i++){
				// Use tensors of about 1MB each
				Tensor t1 = new Tensor(1024*1024/4);
				t1.fill(1.0f);
				
				Tensor t2 = new Tensor(1024*1024/4);
				t2.fill(1.0f);
				
				Tensor res = TensorOps.add(null, t1, t2);
				keep.add(res);
			}
		});
		t.start();
		t.join();
	}

}
