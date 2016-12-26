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
package be.iminds.iot.dianne.dataset;

import java.util.ArrayList;
import java.util.List;

import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import be.iminds.iot.dianne.api.dataset.Batch;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.api.dataset.Sequence;
import be.iminds.iot.dianne.tensor.NativeTensorLoader;
import be.iminds.iot.dianne.tensor.Tensor;

public class SampleTest {
	
	@BeforeClass
	public static void setup() {
		NativeTensorLoader loader = new NativeTensorLoader();
		loader.activate(null);
	}
	
	
	@Test
	public void testSample() throws Exception {
		Tensor input = new Tensor(new float[]{2,2}, 2);
		Tensor target = new Tensor(new float[]{0,1}, 2);

		// sample creation
		Sample s = new Sample(input, target);
		Assert.assertEquals(input, s.input);
		Assert.assertEquals(target, s.target);
		
		// sample cloning
		Sample s2 = s.clone();
		Assert.assertTrue(s2 != s);
		Assert.assertTrue(s2.input != s.input);
		Assert.assertTrue(s2.target != s.target);
		Assert.assertEquals(input, s2.input);
		Assert.assertEquals(target, s2.target);
		
		// sample copyinto
		Tensor input2 = new Tensor(new float[]{0,0}, 2);
		Tensor target2 = new Tensor(new float[]{0,0}, 2);
		Sample s3 = new Sample(input2, target2);
		Assert.assertEquals(input2, s3.input);
		Assert.assertEquals(target2, s3.target);		
		
		s.copyInto(s3);
		Assert.assertEquals(input, s3.input);
		Assert.assertEquals(target, s3.target);
		Assert.assertEquals(input2, s3.input);
		Assert.assertEquals(target2, s3.target);
		
		Sample s4 = new Sample(new Tensor(2), new Tensor(2));
		s.copyInto(s4);
		Assert.assertEquals(input, s4.input);
		Assert.assertEquals(target, s4.target);
		
		Sample s5 = new Sample();
		s.copyInto(s5);
		Assert.assertEquals(input, s5.input);
		Assert.assertEquals(target, s5.target);
	}
	
	
	@Test
	public void testBatch() throws Exception {
		Tensor input = new Tensor(new float[]{2,2, 3, 3},2, 2);
		Tensor target = new Tensor(new float[]{0,1, 1, 0},2, 2);
		
		Tensor input0 = new Tensor(new float[]{2,2}, 2);
		Tensor input1 = new Tensor(new float[]{3,3}, 2);

		Tensor target0 = new Tensor(new float[]{0,1}, 2);
		Tensor target1 = new Tensor(new float[]{1,0}, 2);

		
		// batch creation
		Batch b = new Batch(input, target);
		Assert.assertEquals(input, b.input);
		Assert.assertEquals(target, b.target);
		Assert.assertEquals(input0, b.getInput(0));
		Assert.assertEquals(input1, b.getInput(1));
		Assert.assertEquals(target0, b.getTarget(0));
		Assert.assertEquals(target1, b.getTarget(1));
		
		Assert.assertEquals(input0, b.samples[0].input);
		Assert.assertEquals(input1, b.samples[1].input);
		Assert.assertEquals(target0, b.samples[0].target);
		Assert.assertEquals(target1, b.samples[1].target);
		
		// batch clone
		Batch b2 = b.clone();
		Assert.assertTrue(b2 != b);
		Assert.assertTrue(b2.input != b.input);
		Assert.assertTrue(b2.target != b.target);
		
		Assert.assertEquals(input, b2.input);
		Assert.assertEquals(target, b2.target);
		Assert.assertEquals(input0, b2.getInput(0));
		Assert.assertEquals(input1, b2.getInput(1));
		Assert.assertEquals(target0, b2.getTarget(0));
		Assert.assertEquals(target1, b2.getTarget(1));
		
		Assert.assertEquals(input0, b2.samples[0].input);
		Assert.assertEquals(input1, b2.samples[1].input);
		Assert.assertEquals(target0, b2.samples[0].target);
		Assert.assertEquals(target1, b2.samples[1].target);
		
		// batch copyInto
		Batch b3 = new Batch(2, new int[]{2,2}, new int[]{2,2});
		b.copyInto(b3);
		
		Assert.assertTrue(b3 != b);
		Assert.assertTrue(b3.input != b.input);
		Assert.assertTrue(b3.target != b.target);
		
		Assert.assertEquals(input, b3.input);
		Assert.assertEquals(target, b3.target);
		Assert.assertEquals(input0, b3.getInput(0));
		Assert.assertEquals(input1, b3.getInput(1));
		Assert.assertEquals(target0, b3.getTarget(0));
		Assert.assertEquals(target1, b3.getTarget(1));
		
		Assert.assertEquals(input0, b3.samples[0].input);
		Assert.assertEquals(input1, b3.samples[1].input);
		Assert.assertEquals(target0, b3.samples[0].target);
		Assert.assertEquals(target1, b3.samples[1].target);
	}


	@Test
	public void testSampleSequence() throws Exception {
		Sample s0 = new Sample(new Tensor(new float[]{0,0}, 2), new Tensor(new float[]{0},1));
		Sample s1 = new Sample(new Tensor(new float[]{1,1}, 2), new Tensor(new float[]{1},1));
		Sample s2 = new Sample(new Tensor(new float[]{2,2}, 2), new Tensor(new float[]{2},1));
		Sample s3 = new Sample(new Tensor(new float[]{3,3}, 2), new Tensor(new float[]{3},1));
	
		Sequence<Sample> seq1 = new Sequence<>();
		List<Sample> data = new ArrayList<>();
		data.add(s0);
		data.add(s1);
		data.add(s2);
		Sequence<Sample> seq2 = new Sequence<>(data);
		data.add(s3);
		Sequence<Sample> seq3 = new Sequence<>(data);
		Sequence<Sample> seq4 = new Sequence<>(data, 3);
		
		Assert.assertEquals(0, seq1.size());
		Assert.assertEquals(3, seq2.size());
		Assert.assertEquals(4, seq3.size());
		Assert.assertEquals(3, seq4.size());
		
		Assert.assertEquals(new Tensor(new float[]{0,0},2), seq2.getInput(0));
		Assert.assertEquals(new Tensor(new float[]{1,1},2), seq2.getInput(1));
		Assert.assertEquals(new Tensor(new float[]{2},1), seq2.getTarget(2));

		try {
			seq2.get(3);
			Assert.fail();
		} catch(ArrayIndexOutOfBoundsException e){}

		Sample s = seq3.get(3);
		Assert.assertEquals(new Tensor(new float[]{3,3},2), s.input);
		Assert.assertEquals(new Tensor(new float[]{3},1), s.target);
		
		try {
			seq4.get(3);
			Assert.fail();
		} catch(ArrayIndexOutOfBoundsException e){}
		
		
		seq3.copyInto(seq2);
		Assert.assertEquals(4, seq2.size());
		Assert.assertEquals(new Tensor(new float[]{3,3},2), seq2.getInput(3));

		seq4.copyInto(seq3);
		Assert.assertEquals(new Tensor(new float[]{0,0},2), seq3.getInput(0));
		Assert.assertEquals(new Tensor(new float[]{1,1},2), seq3.getInput(1));
		Assert.assertEquals(new Tensor(new float[]{2},1), seq3.getTarget(2));
		try {
			seq3.get(3);
			Assert.fail();
		} catch(ArrayIndexOutOfBoundsException e){}
	}
	
	
	@Test
	public void testBatchSequence() throws Exception {
		Batch b0 = new Batch(new Tensor(new float[]{0,0,0,0},2, 2), new Tensor(new float[]{0,0},2,1));
		Batch b1 = new Batch(new Tensor(new float[]{1,1,1,1},2, 2), new Tensor(new float[]{1,1},2,1));
		Batch b2 = new Batch(new Tensor(new float[]{2,2,2,2},2, 2), new Tensor(new float[]{2,2},2,1));
		Batch b3 = new Batch(new Tensor(new float[]{3,3,3,3},2, 2), new Tensor(new float[]{3,3},2,1));
	
		Sequence<Batch> seq1 = new Sequence<>();
		List<Batch> data = new ArrayList<>();
		data.add(b0);
		data.add(b1);
		data.add(b2);
		Sequence<Batch> seq2 = new Sequence<>(data);
		data.add(b3);
		Sequence<Batch> seq3 = new Sequence<>(data);
		Sequence<Batch> seq4 = new Sequence<>(data, 3);
		
		Assert.assertEquals(0, seq1.size());
		Assert.assertEquals(3, seq2.size());
		Assert.assertEquals(4, seq3.size());
		Assert.assertEquals(3, seq4.size());
		
		Assert.assertEquals(new Tensor(new float[]{0,0,0,0},2,2), seq2.getInput(0));
		Assert.assertEquals(new Tensor(new float[]{1,1},2), seq2.get(1).getInput(0));
		Assert.assertEquals(new Tensor(new float[]{2,2},2,1), seq2.getTarget(2));

		try {
			seq2.get(3);
			Assert.fail();
		} catch(ArrayIndexOutOfBoundsException e){}

		Batch b = seq3.get(3);
		Assert.assertEquals(new Tensor(new float[]{3,3,3,3},2,2), b.input);
		Assert.assertEquals(new Tensor(new float[]{3,3},2,1), b.target);
		
		try {
			seq4.get(3);
			Assert.fail();
		} catch(ArrayIndexOutOfBoundsException e){}
		
		
		seq3.copyInto(seq2);
		Assert.assertEquals(4, seq2.size());
		Assert.assertEquals(new Tensor(new float[]{3,3,3,3},2,2), seq2.getInput(3));

		seq4.copyInto(seq3);
		Assert.assertEquals(new Tensor(new float[]{0,0,0,0},2,2), seq3.getInput(0));
		Assert.assertEquals(new Tensor(new float[]{1,1},2), seq3.get(1).getInput(0));
		Assert.assertEquals(new Tensor(new float[]{2,2},2,1), seq3.getTarget(2));;
		try {
			seq3.get(3);
			Assert.fail();
		} catch(ArrayIndexOutOfBoundsException e){}
	}
}
