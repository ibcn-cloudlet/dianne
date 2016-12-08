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
 *     Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.nn.learn.criterion;

import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;

import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.tensor.NativeTensorLoader;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;
import junit.framework.Assert;

@Ignore
public abstract class AbstractCriterionTest {
	
	private static final double eps = 1e-5;
	
	protected float[][] out;
	protected float[][] tar;
	protected float[] loss;
	protected float[][] grad;

	@BeforeClass
	public static void loadTensorLibrary() {
		NativeTensorLoader loader = new NativeTensorLoader();
		loader.activate(null);
	}

	@Test
	public void testLoss() {
		for(int i = 0; i < out.length; i++) {
			Criterion c = newCriterion();
			
			Tensor o = new Tensor(out[i],out[i].length);
			Tensor t = new Tensor(tar[i],tar[i].length);
			
			float l = TensorOps.mean(c.loss(o, t));
			
			Assert.assertEquals(loss[i], l, eps);
		}
	}
	
	@Test
	public void testGrad() {
		for(int i = 0; i < out.length; i++) {
			Criterion c = newCriterion();
			
			Tensor o = new Tensor(out[i],out[i].length);
			Tensor t = new Tensor(tar[i],tar[i].length);
			
			c.loss(o, t);
			Tensor g = c.grad(o, t);
			
			for(int j = 0; j < grad[i].length; j++)
				Assert.assertEquals(grad[i][j], g.get(j), eps);
		}
	}
	
	protected abstract Criterion newCriterion();
}
