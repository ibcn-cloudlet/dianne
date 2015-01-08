package be.iminds.iot.dianne.tensor;

import junit.framework.Assert;

import org.junit.Before;
import org.junit.Test;

import be.iminds.iot.dianne.tensor.impl.java.JavaTensorFactory;

public class TensorTest {

	TensorFactory<?> factory;
	
    @Before
    public void setUp() {
        factory = new JavaTensorFactory();
    }
	
	@Test
	public void test1DTensor() {
		Tensor<?> t = factory.createTensor(4);
		Assert.assertEquals(1, t.dim());
		Assert.assertEquals(4, t.size());
		Assert.assertEquals(4, t.size(0));

		Assert.assertEquals(0.0f, t.get(1));
		t.set(1.0f, 1);
		Assert.assertEquals(1.0f, t.get(1));
	}

	@Test
	public void test2DTensor() {
		Tensor<?> t = factory.createTensor(3,4);
		Assert.assertEquals(2, t.dim());
		Assert.assertEquals(12, t.size());
		Assert.assertEquals(3, t.size(0));
		Assert.assertEquals(4, t.size(1));


		Assert.assertEquals(0.0f, t.get(1,2));
		t.set(1.0f, 1,2);
		Assert.assertEquals(1.0f, t.get(1,2));
		
		System.out.println(t);
	}
	
	@Test
	public void test3DTensor() {
		Tensor<?> t = factory.createTensor(2,3,4);
		Assert.assertEquals(3, t.dim());
		Assert.assertEquals(24, t.size());
		Assert.assertEquals(2, t.size(0));
		Assert.assertEquals(3, t.size(1));
		Assert.assertEquals(4, t.size(2));

		Assert.assertEquals(0.0f, t.get(1,2,3));
		t.set(1.0f, 1,2,3);
		Assert.assertEquals(1.0f, t.get(1,2,3));
		
	}
}
