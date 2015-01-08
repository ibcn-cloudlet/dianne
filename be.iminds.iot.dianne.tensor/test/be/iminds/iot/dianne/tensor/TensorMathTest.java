package be.iminds.iot.dianne.tensor;

import static org.junit.Assert.*;
import junit.framework.Assert;

import org.junit.Before;
import org.junit.Test;

import be.iminds.iot.dianne.tensor.impl.java.JavaTensorFactory;

public class TensorMathTest {

	TensorFactory factory;
	TensorMath math;
	
    @Before
    public void setUp() {
        factory = new JavaTensorFactory();
        math = factory.getTensorMath();
    }

	@Test
	public void testAdd1() {
		Tensor t1 = factory.createTensor(2,2);
		t1.fill(2);
		
		Tensor r = math.add(null, t1, 3);
		
		Tensor exp = factory.createTensor(2,2);
		exp.fill(5);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testAdd2() {
		Tensor t1 = factory.createTensor(2,2);
		t1.fill(2);
		Tensor t2 = factory.createTensor(4);
		t2.fill(3);
		
		Tensor r = math.add(null, t1, t2);
		
		Tensor exp = factory.createTensor(2,2);
		exp.fill(5);
		
		Assert.assertEquals(exp, r);
	}

	@Test
	public void testAdd3() {
		Tensor t1 = factory.createTensor(2,2);
		t1.fill(2);
		Tensor t2 = factory.createTensor(4);
		t2.fill(3);
		
		Tensor r = math.add(null, t1, 2, t2);
		
		Tensor exp = factory.createTensor(2,2);
		exp.fill(8);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testMul() {
		Tensor t1 = factory.createTensor(2,2);
		t1.fill(2);
		
		Tensor r = math.mul(null, t1, 2);
		
		Tensor exp = factory.createTensor(2,2);
		exp.fill(4);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testCMul() {
		Tensor t1 = factory.createTensor(2,2);
		t1.fill(2);
		Tensor t2 = factory.createTensor(4);
		t2.fill(3);
		
		Tensor r = math.cmul(null, t1, t2);
		
		Tensor exp = factory.createTensor(2,2);
		exp.fill(6);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testDiv() {
		Tensor t1 = factory.createTensor(2,2);
		t1.fill(6);
		
		Tensor r = math.div(null, t1, 2);
		
		Tensor exp = factory.createTensor(2,2);
		exp.fill(3);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testCDiv() {
		Tensor t1 = factory.createTensor(2,2);
		t1.fill(6);
		Tensor t2 = factory.createTensor(4);
		t2.fill(3);
		
		Tensor r = math.cdiv(null, t1, t2);
		
		Tensor exp = factory.createTensor(2,2);
		exp.fill(2);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testDot() {
		Tensor t1 = factory.createTensor(2,2);
		t1.fill(2);
		Tensor t2 = factory.createTensor(4);
		t2.fill(3);
		
		float dot = math.dot(t1, t2);
		
		Assert.assertEquals(24f, dot);
	}
	
	@Test
	public void testMv1() {
		Tensor t1 = factory.createTensor(2,2);
		t1.fill(2);
		Tensor t2 = factory.createTensor(2);
		t2.fill(3);
		
		Tensor r = math.mv(null, t1, t2);
		
		Tensor exp = factory.createTensor(2);
		exp.fill(12);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testMm1() {
		Tensor t1 = factory.createTensor(2,2);
		t1.fill(2);
		Tensor t2 = factory.createTensor(2,2);
		t2.fill(3);
		
		Tensor r = math.mm(null, t1, t2);
		
		Tensor exp = factory.createTensor(2,2);
		exp.fill(12);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testMm2() {
		Tensor t1 = factory.createTensor(3,2);
		for(int i=0;i<6;i++){
			t1.set(i+1, i);
		}
		Tensor t2 = factory.createTensor(2,3);
		for(int i=0;i<6;i++){
			t2.set(i+1, i);
		}
	
		Tensor r = math.mm(null, t1, t2);
		
		Tensor exp = factory.createTensor(3,3);
		exp.set(9, 0);
		exp.set(12, 1);
		exp.set(15, 2);
		exp.set(19, 3);
		exp.set(26, 4);
		exp.set(33, 5);
		exp.set(29, 6);
		exp.set(40, 7);
		exp.set(51, 8);
		
		Assert.assertEquals(exp, r);
	}
}
