package be.iminds.iot.dianne.nn.learn.criterion;

import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;

import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.tensor.NativeTensorLoader;
import be.iminds.iot.dianne.tensor.Tensor;
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
			
			float l = c.loss(
					new Tensor(out[i],out[i].length),
					new Tensor(tar[i],tar[i].length));
			
			Assert.assertEquals(loss[i], l, eps);
		}
	}
	
	@Test
	public void testGrad() {
		for(int i = 0; i < out.length; i++) {
			Criterion c = newCriterion();
			
			Tensor g = c.grad(
					new Tensor(out[i],out[i].length),
					new Tensor(tar[i],tar[i].length));
			
			for(int j = 0; j < grad[i].length; j++)
				Assert.assertEquals(grad[i][j], g.get(j), eps);
		}
	}
	
	protected abstract Criterion newCriterion();
}
