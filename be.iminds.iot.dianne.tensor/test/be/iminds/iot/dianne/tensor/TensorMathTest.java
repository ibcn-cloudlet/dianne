package be.iminds.iot.dianne.tensor;

import junit.framework.Assert;

import org.junit.Before;
import org.junit.Test;

import be.iminds.iot.dianne.tensor.impl.java.JavaTensor;
import be.iminds.iot.dianne.tensor.impl.java.JavaTensorMath;

public class TensorMathTest<T extends Tensor<T>> {

	TensorFactory<T> factory;
	TensorMath<T> math;
	
    @Before
    public void setUp() {
        factory = new TensorFactory(JavaTensor.class, JavaTensorMath.class);
        math = factory.getTensorMath();
    }

	@Test
	public void testAdd1() {
		T t1 = factory.createTensor(2,2);
		t1.fill(2);
		
		T r = math.add(null, t1, 3);
		
		T exp = factory.createTensor(2,2);
		exp.fill(5);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testAdd2() {
		T t1 = factory.createTensor(2,2);
		t1.fill(2);
		T t2 = factory.createTensor(4);
		t2.fill(3);
		
		T r = math.add(null, t1, t2);
		
		T exp = factory.createTensor(2,2);
		exp.fill(5);
		
		Assert.assertEquals(exp, r);
	}

	@Test
	public void testAdd3() {
		T t1 = factory.createTensor(2,2);
		t1.fill(2);
		T t2 = factory.createTensor(4);
		t2.fill(3);
		
		T r = math.add(null, t1, 2, t2);
		
		T exp = factory.createTensor(2,2);
		exp.fill(8);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testSub1() {
		T t1 = factory.createTensor(2,2);
		t1.fill(3);
		
		T r = math.sub(null, t1, 1);
		
		T exp = factory.createTensor(2,2);
		exp.fill(2);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testSub2() {
		T t1 = factory.createTensor(2,2);
		t1.fill(3);
		T t2 = factory.createTensor(4);
		t2.fill(1);
		
		T r = math.sub(null, t1, t2);
		
		T exp = factory.createTensor(2,2);
		exp.fill(2);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testMul() {
		T t1 = factory.createTensor(2,2);
		t1.fill(2);
		
		T r = math.mul(null, t1, 2);
		
		T exp = factory.createTensor(2,2);
		exp.fill(4);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testCMul() {
		T t1 = factory.createTensor(2,2);
		t1.fill(2);
		T t2 = factory.createTensor(4);
		t2.fill(3);
		
		T r = math.cmul(null, t1, t2);
		
		T exp = factory.createTensor(2,2);
		exp.fill(6);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testDiv() {
		T t1 = factory.createTensor(2,2);
		t1.fill(6);
		
		T r = math.div(null, t1, 2);
		
		T exp = factory.createTensor(2,2);
		exp.fill(3);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testCDiv() {
		T t1 = factory.createTensor(2,2);
		t1.fill(6);
		T t2 = factory.createTensor(4);
		t2.fill(3);
		
		T r = math.cdiv(null, t1, t2);
		
		T exp = factory.createTensor(2,2);
		exp.fill(2);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testDot() {
		T t1 = factory.createTensor(2,2);
		t1.fill(2);
		T t2 = factory.createTensor(4);
		t2.fill(3);
		
		float dot = math.dot(t1, t2);
		
		Assert.assertEquals(24f, dot);
	}
	
	@Test
	public void testMv1() {
		T t1 = factory.createTensor(2,2);
		t1.fill(2);
		T t2 = factory.createTensor(2);
		t2.fill(3);
		
		T r = math.mv(null, t1, t2);
		
		T exp = factory.createTensor(2);
		exp.fill(12);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testMm1() {
		T t1 = factory.createTensor(2,2);
		t1.fill(2);
		T t2 = factory.createTensor(2,2);
		t2.fill(3);
		
		T r = math.mm(null, t1, t2);
		
		T exp = factory.createTensor(2,2);
		exp.fill(12);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testMm2() {
		T t1 = factory.createTensor(3,2);
		int k = 1; 
		for(int i=0;i<3;i++){
			for(int j=0;j<2;j++){
				t1.set(k, i, j);
				k++;
			}
		}
		
		T t2 = factory.createTensor(2,3);
		k = 1; 
		for(int i=0;i<2;i++){
			for(int j=0;j<3;j++){
				t2.set(k, i, j);
				k++;
			}
		}

		T r = math.mm(null, t1, t2);
		
		T exp = factory.createTensor(3,3);
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
	public void testSum() {
		T t1 = factory.createTensor(4);
		t1.fill(2);
		
		Assert.assertEquals(8.0, math.sum(t1), 0.1f);
	}
	
	@Test
	public void testMax() {
		T t1 = factory.createTensor(4);
		t1.fill(2);
		t1.set(0, 0);
		t1.set(5, 3);
		
		Assert.assertEquals(5.0, math.max(t1), 0.1f);
	}
	
	@Test
	public void testMin() {
		T t1 = factory.createTensor(4);
		t1.fill(2);
		t1.set(0, 0);
		t1.set(5, 3);
		
		Assert.assertEquals(0.0, math.min(t1), 0.1f);
	}
	
	@Test
	public void testMean() {
		T t1 = factory.createTensor(4);
		t1.fill(2);
		t1.set(0, 0);
		t1.set(5, 3);
		
		Assert.assertEquals(2.25, math.mean(t1), 0.1f);
	}
}
