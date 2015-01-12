package be.iminds.iot.dianne.tensor;

import junit.framework.Assert;

import org.junit.Before;
import org.junit.Test;

import be.iminds.iot.dianne.tensor.impl.java.JavaTensor;
import be.iminds.iot.dianne.tensor.impl.java.JavaTensorMath;

public class TensorTest<T extends Tensor<T>> {

	TensorFactory<T> factory;
	
    @Before
    public void setUp() {
        factory = new TensorFactory(JavaTensor.class, JavaTensorMath.class);
    }
	
	@Test
	public void test1DTensor() {
		T t = factory.createTensor(4);
		Assert.assertEquals(1, t.dim());
		Assert.assertEquals(4, t.size());
		Assert.assertEquals(4, t.size(0));

		Assert.assertEquals(0.0f, t.get(1));
		t.set(1.0f, 1);
		Assert.assertEquals(1.0f, t.get(1));
	}

	@Test
	public void test2DTensor() {
		T t = factory.createTensor(3,4);
		Assert.assertEquals(2, t.dim());
		Assert.assertEquals(12, t.size());
		Assert.assertEquals(3, t.size(0));
		Assert.assertEquals(4, t.size(1));

		Assert.assertEquals(0.0f, t.get(1,2));
		t.set(1.0f, 1,2);
		Assert.assertEquals(1.0f, t.get(1,2));

	}
	
	@Test
	public void test3DTensor() {
		T t = factory.createTensor(2,3,4);
		Assert.assertEquals(3, t.dim());
		Assert.assertEquals(24, t.size());
		Assert.assertEquals(2, t.size(0));
		Assert.assertEquals(3, t.size(1));
		Assert.assertEquals(4, t.size(2));

		Assert.assertEquals(0.0f, t.get(1,2,3));
		t.set(1.0f, 1,2,3);
		Assert.assertEquals(1.0f, t.get(1,2,3));
		
	}
	
	@Test
	public void testEquals() {
		T t = factory.createTensor(2,2);
		T t2 = factory.createTensor(2,2);
		T t3 = factory.createTensor(4);
		
		Assert.assertEquals(true, t.equals(t2));
		Assert.assertEquals(false, t.equals(t3));
		t.set(1.0f, 0, 0);
		Assert.assertEquals(false, t.equals(t2));

	}
	
	@Test
	public void testClone() {
		T t = factory.createTensor(2,2);
		t.set(0f, 0, 0);
		t.set(1f, 0, 1);
		t.set(2f, 1, 0);
		t.set(3f, 1, 1);
		
		T t2 = t.clone(null);
		Assert.assertEquals(0f, t2.get(0, 0), 0.1f);
		Assert.assertEquals(1f, t2.get(0, 1), 0.1f);
		Assert.assertEquals(2f, t2.get(1, 0), 0.1f);
		Assert.assertEquals(3f, t2.get(1, 1), 0.1f);
		
		T t3 = factory.createTensor(4);
		t.clone(t3);
		
		Assert.assertEquals(0f, t3.get(0), 0.1f);
		Assert.assertEquals(1f, t3.get(1), 0.1f);
		Assert.assertEquals(2f, t3.get(2), 0.1f);
		Assert.assertEquals(3f, t3.get(3), 0.1f);
	}
	
	@Test
	public void testTranspose() {
		T t = factory.createTensor(2,3);
		for(int i = 0; i < 2; i++)
			for(int j = 0; j < 3; j++)
				t.set(i*2+j, i, j);
		
		T t2 = t.transpose(null, 0, 1);
		for(int i = 0; i < 3; i++)
			for(int j = 0; j < 2; j++)
				Assert.assertEquals(j*2+i, t2.get(i, j), 0.1f);
	}
	
	@Test
	public void testNarrow1() {
		T t = factory.createTensor(5,6);
		
		T t2 = t.narrow(0,1,3);
		t2.set(1.0f, 0,2);
		
		for(int i = 0; i < 5; i++)
			for(int j = 0; j < 6; j++){
				if(i==1 && j==2)
					Assert.assertEquals(1.0f, t.get(i, j), 0.1f);
				else
					Assert.assertEquals(0.0f, t.get(i, j), 0.1f);
			}
	}
	
	@Test
	public void testNarrow2() {
		T t = factory.createTensor(5,6);
		
		T t2 = t.narrow(0,1,3);
		t2.fill(1);
		
		for(int i = 0; i < 3; i++)
			for(int j = 0; j < 6; j++)
				Assert.assertEquals(1.0f, t2.get(i, j), 0.1f);
		
		for(int i = 0; i < 5; i++)
			for(int j = 0; j < 6; j++){
				if(i==0 || i==4)
					Assert.assertEquals(0.0f, t.get(i, j), 0.1f);
				else
					Assert.assertEquals(1.0f, t.get(i, j), 0.1f);
			}
	}
	
	@Test
	public void testNarrow3() {
		T t = factory.createTensor(5,6);
		
		T t2 = t.narrow(1,3,2,4);
		t2.fill(1);
	
		for(int i = 0; i < 3; i++)
			for(int j = 0; j < 3; j++)
				Assert.assertEquals(1.0f, t2.get(i, j), 0.1f);
		
		for(int i = 0; i < 5; i++)
			for(int j = 0; j < 6; j++){
				if(i >= 1 && i<=3 && j >= 2 && j <=4)
					Assert.assertEquals(1.0f, t.get(i, j), 0.1f);
				else
					Assert.assertEquals(0.0f, t.get(i, j), 0.1f);
			}
	}
	
	@Test
	public void testCloneNarrow() {
		T t = factory.createTensor(2,3);
		for(int i = 0; i < 2; i++)
			for(int j = 0; j < 3; j++)
				t.set(i*2+j, i, j);
		
		T t2 = t.narrow(0, 1, 1);
		T t3 = t2.clone(null);
		
		Assert.assertEquals(true, t3.equals(t2));
		Assert.assertEquals(3, t3.size());
	}
}
