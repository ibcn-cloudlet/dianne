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
 * but WITHOUTensor ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez, Elias De Coninck
 *******************************************************************************/
package be.iminds.iot.dianne.tensor;

import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

public class TensorOpsTest {

	@BeforeClass
	public static void load() {
		NativeTensorLoader loader = new NativeTensorLoader();
		loader.activate(null);
	}
	
	@Test
	public void testAdd1() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(2);
		
		Tensor r = TensorOps.add(null, t1, 3);
		
		Tensor exp = new Tensor(2,2);
		exp.fill(5);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testAdd2() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(2);
		Tensor t2 = new Tensor(4);
		t2.fill(3);
		
		Tensor r = TensorOps.add(null, t1, t2);
		
		Tensor exp = new Tensor(2,2);
		exp.fill(5);
		
		Assert.assertEquals(exp, r);
	}

	@Test
	public void testAdd3() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(2);
		Tensor t2 = new Tensor(4);
		t2.fill(3);
		
		Tensor r = TensorOps.add(null, t1, 2, t2);
		
		Tensor exp = new Tensor(2,2);
		exp.fill(8);
		
		Assert.assertEquals(exp, r);
	}

	@Test
	public void testAdd4() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(2);
		Tensor t2 = new Tensor(4);
		t2.fill(3);
		
		Tensor r = new Tensor(2,2);
		r.fill(1.0f);
		TensorOps.add(r, t1, 2, t2);
		
		Tensor exp = new Tensor(2,2);
		exp.fill(8);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testSub1() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(3);
		
		Tensor r = TensorOps.sub(null, t1, 1);
		
		Tensor exp = new Tensor(2,2);
		exp.fill(2);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testSub2() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(3);
		Tensor t2 = new Tensor(4);
		t2.fill(1);
		
		Tensor r = TensorOps.sub(null, t1, t2);
		
		Tensor exp = new Tensor(2,2);
		exp.fill(2);
		
		Assert.assertEquals(exp, r);
	}

	@Test
	public void testSub3() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(3);
		Tensor t2 = new Tensor(4);
		t2.fill(1);
		
		Tensor r = new Tensor(2,2);
		r.fill(1.0f);
		TensorOps.sub(r, t1, t2);
		
		Tensor exp = new Tensor(2,2);
		exp.fill(2);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testMul() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(2);
		
		Tensor r = TensorOps.mul(null, t1, 2);
		
		Tensor exp = new Tensor(2,2);
		exp.fill(4);
		
		Assert.assertEquals(exp, r);
	}

	@Test
	public void testMul2() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(2);
		
		Tensor r = new Tensor(2,2);
		r.fill(1.0f);
		TensorOps.mul(r, t1, 2);
		
		Tensor exp = new Tensor(2,2);
		exp.fill(4);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testCMul() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(2);
		Tensor t2 = new Tensor(4);
		t2.fill(3);
		
		Tensor r = TensorOps.cmul(null, t1, t2);
		
		Tensor exp = new Tensor(2,2);
		exp.fill(6);
		
		Assert.assertEquals(exp, r);
	}

	@Test
	public void testCMul2() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(2);
		Tensor t2 = new Tensor(4);
		t2.fill(3);
		
		Tensor r = new Tensor(2,2);
		r.fill(1.0f);
		TensorOps.cmul(r, t1, t2);
		
		Tensor exp = new Tensor(2,2);
		exp.fill(6);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testDiv() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(6);
		
		Tensor r = TensorOps.div(null, t1, 2);
		
		Tensor exp = new Tensor(2,2);
		exp.fill(3);
		
		Assert.assertEquals(exp, r);
	}

	@Test
	public void testDiv2() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(6);
		
		Tensor r = new Tensor(2,2);
		r.fill(1.0f);
		TensorOps.div(r, t1, 2);
		
		Tensor exp = new Tensor(2,2);
		exp.fill(3);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testCDiv() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(6);
		Tensor t2 = new Tensor(4);
		t2.fill(3);
		
		Tensor r = TensorOps.cdiv(null, t1, t2);
		
		Tensor exp = new Tensor(2,2);
		exp.fill(2);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testCDiv2() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(6);
		Tensor t2 = new Tensor(4);
		t2.fill(3);
		
		Tensor r = new Tensor(2,2);
		r.fill(1.0f);
		TensorOps.cdiv(r, t1, t2);
		
		Tensor exp = new Tensor(2,2);
		exp.fill(2);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testDot() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(2);
		Tensor t2 = new Tensor(4);
		t2.fill(3);
		
		float dot = TensorOps.dot(t1, t2);
		
		Assert.assertEquals(24f, dot, 0.01);
	}
	
	@Test
	public void testMv1() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(2);
		Tensor t2 = new Tensor(2);
		t2.fill(3);
		
		Tensor r = TensorOps.mv(null, t1, t2);
		
		Tensor exp = new Tensor(2);
		exp.fill(12);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testMv2() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(2);
		Tensor t2 = new Tensor(2);
		t2.fill(3);
		
		Tensor r = new Tensor(2);
		r.fill(1.0f);
		TensorOps.mv(r, t1, t2);
		
		Tensor exp = new Tensor(2);
		exp.fill(12);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testMm1() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(2);
		Tensor t2 = new Tensor(2,2);
		t2.fill(3);
		
		Tensor r = TensorOps.mm(null, t1, t2);
		
		Tensor exp = new Tensor(2,2);
		exp.fill(12);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testMm2() {
		Tensor t1 = new Tensor(3,2);
		int k = 1; 
		for(int i=0;i<3;i++){
			for(int j=0;j<2;j++){
				t1.set(k, i, j);
				k++;
			}
		}
		
		Tensor t2 = new Tensor(2,3);
		k = 1; 
		for(int i=0;i<2;i++){
			for(int j=0;j<3;j++){
				t2.set(k, i, j);
				k++;
			}
		}

		Tensor r = TensorOps.mm(null, t1, t2);
		
		Tensor exp = new Tensor(3,3);
		exp.set(9, 0, 0);
		exp.set(12, 0, 1);
		exp.set(15, 0, 2);
		exp.set(19, 1, 0);
		exp.set(26, 1, 1);
		exp.set(33, 1, 2);
		exp.set(29, 2, 0);
		exp.set(40, 2, 1);
		exp.set(51, 2, 2);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testMm3() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(2);
		Tensor t2 = new Tensor(2,2);
		t2.fill(3);
		
		Tensor r = new Tensor(2,2);
		r.fill(1.0f);
		TensorOps.mm(r, t1, t2);
		
		Tensor exp = new Tensor(2,2);
		exp.fill(12);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testAddmv() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(2);
		Tensor t2 = new Tensor(2);
		t2.fill(3);
		Tensor t3 = new Tensor(2);
		t3.fill(1.0f);
		
		Tensor r = TensorOps.addmv(null, t3, t1, t2);
		
		Tensor exp = new Tensor(2);
		exp.fill(13);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testAddmv2() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(2);
		Tensor t2 = new Tensor(2);
		t2.fill(3);
		Tensor t3 = new Tensor(2);
		t3.fill(1.0f);
		
		Tensor r = new Tensor(2);
		r.fill(1.0f);
		TensorOps.addmv(r, t3, t1, t2);
		
		Tensor exp = new Tensor(2);
		exp.fill(13);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testSum() {
		Tensor t1 = new Tensor(4);
		t1.fill(2);
		
		Assert.assertEquals(8.0, TensorOps.sum(t1), 0.1f);
	}
	
	@Test
	public void testMax() {
		Tensor t1 = new Tensor(4);
		t1.fill(2);
		t1.set(0, 0);
		t1.set(5, 3);
		
		Assert.assertEquals(5.0, TensorOps.max(t1), 0.1f);
	}
	
	@Test
	public void testMin() {
		Tensor t1 = new Tensor(4);
		t1.fill(2);
		t1.set(1, 1);
		t1.set(5, 3);

		Assert.assertEquals(1.0, TensorOps.min(t1), 0.1f);
	}
	
	@Test
	public void testMean() {
		Tensor t1 = new Tensor(4);
		t1.fill(2);
		t1.set(0, 0);
		t1.set(5, 3);
		
		Assert.assertEquals(2.25, TensorOps.mean(t1), 0.1f);
	}
	
	@Test
	public void testArgMax() {
		Tensor t1 = new Tensor(4);
		t1.fill(2);
		t1.set(0, 0);
		t1.set(5, 3);
		
		Assert.assertEquals(3, TensorOps.argmax(t1));
	}
	
	@Test
	public void testArgMin() {
		Tensor t1 = new Tensor(4);
		t1.fill(2);
		t1.set(0, 0);
		t1.set(5, 3);
		Assert.assertEquals(0, TensorOps.argmin(t1));
	}
	
	@Test
	public void testScaleUp(){
		float[] data = new float[]{0, 0, 0, 0, 1, 0, 0, 0, 0}; 
		Tensor t = new Tensor(data, 3, 3);
		
		float[] expData = new float[]{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.16000001f, 0.32000002f, 0.32f, 0.16f, 0.0f, 0.0f, 0.32000002f, 0.64000005f, 0.64f, 0.32f, 0.0f, 0.0f, 0.32f, 0.64f, 0.6399999f, 0.31999996f, 0.0f, 0.0f, 0.16f, 0.32f, 0.31999996f, 0.15999998f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
		Tensor exp = new Tensor(expData, 6, 6);
		
		Tensor result = TensorOps.scale2D(null, t, new int[]{6, 6});
		Assert.assertEquals(exp, result);
	}
	
	@Test
	public void testScaleUp2(){
		float[] data = new float[]{0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0}; 
		Tensor t = new Tensor(data, 2, 3, 3);
		
		float[] expData = new float[]{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.16000001f, 0.32000002f, 0.32f, 0.16f, 0.0f, 0.0f, 0.32000002f, 0.64000005f, 0.64f, 0.32f, 0.0f, 0.0f, 0.32f, 0.64f, 0.6399999f, 0.31999996f, 0.0f, 0.0f, 0.16f, 0.32f, 0.31999996f, 0.15999998f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
									  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.32000002f, 0.64000005f, 0.64f, 0.32f, 0.0f, 0.0f, 0.64000005f, 1.2800001f, 1.28f, 0.64f, 0.0f, 0.0f, 0.64f, 1.28f, 1.2799999f, 0.6399999f, 0.0f, 0.0f, 0.32f, 0.64f, 0.6399999f, 0.31999996f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

		Tensor exp = new Tensor(expData, 2, 6, 6);
		
		Tensor result = TensorOps.scale2D(null, t, new int[]{2, 6, 6});
		Assert.assertEquals(exp, result);
	}
	
	@Test
	public void testScaleUp3(){
		Tensor t = new Tensor(2, 4, 4);
		t.fill(1.0f);
		
		Tensor exp = new Tensor(2, 15, 15);
		exp.fill(1.0f);
		
		Tensor result = TensorOps.scale2D(null, t, new int[]{2, 15, 15});
		Assert.assertTrue(exp.equals(result, 0.00001f));
	}
	
	@Test
	public void testScaleDown(){
		float[] data = new float[]{0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0}; 
		Tensor t = new Tensor(data, 4, 4);
		
		float[] expData = new float[]{0, 0, 0, 0, 1, 0, 0, 0, 0};
		Tensor exp = new Tensor(expData, 3, 3);
		
		Tensor result = TensorOps.scale2D(null, t, new int[]{3, 3});
		Assert.assertEquals(exp, result);
	}
	
	@Test
	public void testScaleDown2(){
		float[] data = new float[]{0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
				                   0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0}; 
		Tensor t = new Tensor(data, 2, 4, 4);
		
		float[] expData = new float[]{0, 0, 0, 0, 1, 0, 0, 0, 0,
				                      0, 0, 0, 0, 2, 0, 0, 0, 0};
		Tensor exp = new Tensor(expData, 2, 3, 3);
		
		Tensor result = TensorOps.scale2D(null, t, new int[]{2, 3, 3});
		Assert.assertEquals(exp, result);
	}
	
	@Test
	public void testScaleDown3(){
		Tensor t = new Tensor(16, 16);
		t.fill(1.0f);
		
		Tensor exp = new Tensor(3, 4, 4);
		exp.fill(1.0f);
		
		Tensor result = TensorOps.scale2D(null, t, new int[]{3, 4, 4});
		Assert.assertTrue(exp.equals(result, 0.00001f));
	}
	
	@Test
	public void testClamp1() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(2);
		t1.set(-1, 0, 0);
		
		Tensor r = TensorOps.clamp(null, t1, 0, 1);
		
		Tensor exp = new Tensor(2,2);
		exp.fill(1);
		exp.set(0, 0, 0);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testClamp2() {
		Tensor t1 = new Tensor(2,2);
		t1.fill(2);
		t1.set(-1, 0, 0);
		
		Tensor r = new Tensor(2,2);
		TensorOps.clamp(r, t1, 0, 1);
		
		Tensor exp = new Tensor(2,2);
		exp.fill(1);
		exp.set(0, 0, 0);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testClamp3() {
		Tensor exp = new Tensor(2,2);
				
		Tensor t1 = new Tensor(2,2);
		t1.fill(Float.POSITIVE_INFINITY);
		TensorOps.clamp(t1, t1, -2.0f, -1.0f);
		exp.fill(-1.0f);
		Assert.assertEquals(exp, t1);

		TensorOps.clamp(t1, t1, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY);
		Assert.assertEquals(exp, t1);
	}
}
