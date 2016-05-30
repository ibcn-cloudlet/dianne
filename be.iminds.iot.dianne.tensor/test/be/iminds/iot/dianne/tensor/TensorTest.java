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

import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

public class TensorTest {

	@BeforeClass
	public static void load() {
		NativeTensorLoader loader = new NativeTensorLoader();
		loader.activate(null);
	}
	
	@Test
	public void test1DTensor() {
		Tensor t = new Tensor(4);
		t.fill(0.0f);
		Assert.assertEquals(1, t.dim());
		Assert.assertEquals(4, t.size());
		Assert.assertEquals(4, t.size(0));

		Assert.assertTrue(t.hasDim(new int[]{4}));
		
		Assert.assertEquals(0.0f, t.get(1), 0.01);
		t.set(1.0f, 1);
		Assert.assertEquals(1.0f, t.get(1), 0.01);
	}

	@Test
	public void test2DTensor() {
		Tensor t = new Tensor(3, 4);
		t.fill(0.0f);
		Assert.assertEquals(2, t.dim());
		Assert.assertEquals(12, t.size());
		Assert.assertEquals(3, t.size(0));
		Assert.assertEquals(4, t.size(1));

		Assert.assertTrue(t.hasDim(new int[]{3, 4}));
		
		Assert.assertEquals(0.0f, t.get(1, 2), 0.01);
		t.set(1.0f, 1, 2);
		Assert.assertEquals(1.0f, t.get(1, 2), 0.01);

	}

	@Test
	public void test2DTensor2() {
		float[] data = new float[]{1.0f, 2.0f, 3.0f, 4.0f};
		Tensor t = new Tensor(data, 2, 2);
		Assert.assertEquals(2, t.dim());
		Assert.assertEquals(4, t.size());
		Assert.assertEquals(2, t.size(0));
		Assert.assertEquals(2, t.size(1));
		
		Assert.assertTrue(t.hasDim(new int[]{2, 2}));

		Assert.assertEquals(1.0f, t.get(0, 0), 0.01);
		
		float[] newdata = t.get();
		Assert.assertArrayEquals(data, newdata, 0.01f);

	}
	
	@Test
	public void test3DTensor() {
		Tensor t = new Tensor(2, 3, 4);
		t.fill(0.0f);
		Assert.assertEquals(3, t.dim());
		Assert.assertEquals(24, t.size());
		Assert.assertEquals(2, t.size(0));
		Assert.assertEquals(3, t.size(1));
		Assert.assertEquals(4, t.size(2));

		Assert.assertEquals(0.0f, t.get(1, 2, 3), 0.01);
		t.set(1.0f, 1, 2, 3);
		Assert.assertEquals(1.0f, t.get(1, 2, 3), 0.01);

	}

	@Test
	public void testEquals() {
		Tensor t = new Tensor(2, 2);
		t.fill(1.0f);
		Tensor t2 = new Tensor(2, 2);
		t2.fill(1.0f);
		Tensor t3 = new Tensor(4);
		t3.fill(1.0f);

		Assert.assertEquals(true, t.equals(t2));
		Assert.assertEquals(false, t.equals(t3));
		t.set(2.0f, 0, 0);
		Assert.assertEquals(false, t.equals(t2));
	}
	
	@Test
	public void testFill(){
		Tensor t = new Tensor(2, 2);
		t.fill(1.0f);
		Tensor expected = new Tensor(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2);
		
		Assert.assertEquals(expected, t);
	}
	
	@Test
	public void testReshape() {
		Tensor t = new Tensor(2, 2);
		t.fill(1.0f);
		Tensor t2 = new Tensor(4);
		t2.fill(1.0f);

		t.reshape(4);
		Assert.assertEquals(t2, t);
		Assert.assertTrue(t.sameDim(t2));
	}

	@Test
	public void testCopyInto() {
		Tensor t = new Tensor(2, 2);
		t.set(0f, 0, 0);
		t.set(1f, 0, 1);
		t.set(2f, 1, 0);
		t.set(3f, 1, 1);

		Tensor t2 = t.copyInto(null);
		Assert.assertEquals(0f, t2.get(0, 0), 0.1f);
		Assert.assertEquals(1f, t2.get(0, 1), 0.1f);
		Assert.assertEquals(2f, t2.get(1, 0), 0.1f);
		Assert.assertEquals(3f, t2.get(1, 1), 0.1f);

		Tensor t3 = new Tensor(4);
		t.copyInto(t3);
		
		Assert.assertEquals(0f, t3.get(0), 0.1f);
		Assert.assertEquals(1f, t3.get(1), 0.1f);
		Assert.assertEquals(2f, t3.get(2), 0.1f);
		Assert.assertEquals(3f, t3.get(3), 0.1f);
	}

	@Test
	public void testTranspose() {
		Tensor t = new Tensor(2, 3);
		for (int i = 0; i < 2; i++)
			for (int j = 0; j < 3; j++)
				t.set(i * 2 + j, i, j);

		Tensor t2 = t.transpose(null, 0, 1);
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 2; j++)
				Assert.assertEquals(j * 2 + i, t2.get(i, j), 0.1f);
	}

	@Test
	public void testNarrow1() {
		Tensor t = new Tensor(5, 6);
		t.fill(0.0f);
		
		Tensor t2 = t.narrow(0, 1, 3);
		t2.set(1.0f, 0, 2);

		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 6; j++) {
				if (i == 1 && j == 2)
					Assert.assertEquals(1.0f, t.get(i, j), 0.1f);
				else
					Assert.assertEquals(0.0f, t.get(i, j), 0.1f);
			}
	}

	@Test
	public void testNarrow2() {
		Tensor t = new Tensor(5, 6);
		t.fill(0.0f);
		Tensor t2 = t.narrow(0, 1, 3);
		t2.fill(1);

		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 6; j++)
				Assert.assertEquals(1.0f, t2.get(i, j), 0.1f);

		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 6; j++) {
				if (i == 0 || i == 4)
					Assert.assertEquals(0.0f, t.get(i, j), 0.1f);
				else
					Assert.assertEquals(1.0f, t.get(i, j), 0.1f);
			}
	}

	@Test
	public void testNarrow3() {
		Tensor t = new Tensor(5, 6);
		t.fill(0.0f);
		Tensor t2 = t.narrow(1, 3, 2, 3);
		t2.fill(1);

		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				Assert.assertEquals(1.0f, t2.get(i, j), 0.1f);

		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 6; j++) {
				if (i >= 1 && i <= 3 && j >= 2 && j <= 4)
					Assert.assertEquals(1.0f, t.get(i, j), 0.1f);
				else
					Assert.assertEquals(0.0f, t.get(i, j), 0.1f);
			}
	}

	@Test
	public void testNarrowSelect() {
		float[] data = new float[3*5*5];
		for(int i=0;i<data.length;i++){
			data[i] = i;
		}
		Tensor t = new Tensor(data, 3, 5, 5);
		
		t = t.narrow(0, 3, 1, 2, 1, 3);
		t = t.select(0, 0);
		
		float[] ed = new float[]{6.0f, 7.0f, 8.0f, 11.0f, 12.0f, 13.0f};
		Tensor exp = new Tensor(ed, 2,3);
		Assert.assertEquals(exp, t);
	}
	
	@Test
	public void testCopyIntoNarrow() {
		Tensor t = new Tensor(2, 3);
		for (int i = 0; i < 2; i++)
			for (int j = 0; j < 3; j++)
				t.set(i * 2 + j, i, j);

		Tensor t2 = t.narrow(0, 1, 1);
		Tensor t3 = t2.copyInto(null);

		Assert.assertEquals(true, t3.equals(t2));
		Assert.assertEquals(3, t3.size());
	}

	@Test
	public void testSelect() {
		Tensor t = new Tensor(3, 3);
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				t.set(i * 2 + j, i, j);

		Tensor row1 = t.select(0, 1);
		Tensor expected = new Tensor(3);
		expected.set(2.0f, 0);
		expected.set(3.0f, 1);
		expected.set(4.0f, 2);

		Assert.assertEquals(1, row1.dim());
		Assert.assertEquals(3, row1.dims()[0]);
		Assert.assertEquals(expected, row1);

		Tensor col1 = t.select(1, 1);

		expected.set(1.0f, 0);
		expected.set(3.0f, 1);
		expected.set(5.0f, 2);

		Assert.assertEquals(1, col1.dim());
		Assert.assertEquals(3, col1.dims()[0]);
		Assert.assertEquals(expected, col1);
	}

	@Test
	public void testDiag() {
		Tensor t = new Tensor(3, 3);
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				t.set(i * 2 + j, i, j);

		Tensor diag = t.diag(null);
		Tensor expected = new Tensor(3);
		expected.set(0.0f, 0);
		expected.set(3.0f, 1);
		expected.set(6.0f, 2);
		Assert.assertEquals(expected, diag);
	}
}
