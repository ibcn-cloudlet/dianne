package be.iminds.iot.dianne.tensor.impl.java;

import junit.framework.Assert;

import org.junit.Test;

public class JavaTensorTest {

	@Test
	public void testJavaTensorConstructor1() {
		JavaTensor tensor = new JavaTensor(2,2);
		Assert.assertEquals(2, tensor.dims.length);
		Assert.assertEquals(2, tensor.strides.length);
		Assert.assertEquals(4, tensor.data.length);
	}

	
	@Test
	public void testJavaTensorConstructor2() {
		float[] data = new float[]{1.0f, 2.0f, 3.0f, 4.0f};
		JavaTensor tensor = new JavaTensor(new int[]{2,2}, data);
		
		Assert.assertEquals(2, tensor.dims.length);
		Assert.assertEquals(2, tensor.strides.length);
		Assert.assertEquals(4, tensor.data.length);
		
		for(int i=0;i<data.length;i++){
			Assert.assertEquals(data[i], tensor.data[i]);
		}
	}
	
	@Test
	public void testJavaTensorStride() {
		JavaTensor tensor = new JavaTensor(3,4);
		tensor.set(1.0f, 1,2);
		Assert.assertEquals(1.0f, tensor.data[6]);
	}
	
	@Test
	public void testJavaTensorToString1() {
		JavaTensor tensor = new JavaTensor(2,2);
		String expected = "[0.0, 0.0]\n[0.0, 0.0]\n";
		Assert.assertEquals(expected, tensor.toString());
	}

	@Test
	public void testJavaTensorToString2() {
		float[] data = new float[]{1.0f, 2.0f, 3.0f, 4.0f};
		JavaTensor tensor = new JavaTensor(new int[]{2,2}, data);
		String expected = "[1.0, 2.0]\n[3.0, 4.0]\n";
		Assert.assertEquals(expected, tensor.toString());
	}
	
	@Test
	public void testJavaTensorToString3() {
		float[] data = new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
		JavaTensor tensor = new JavaTensor(new int[]{2,3}, data);
		String expected = "[1.0, 2.0, 3.0]\n[4.0, 5.0, 6.0]\n";
		Assert.assertEquals(expected, tensor.toString());
	}
	
	@Test
	public void testJavaTensorToString4() {
		JavaTensor tensor = new JavaTensor(3);
		String expected = "[0.0, 0.0, 0.0]";
		Assert.assertEquals(expected, tensor.toString());
	}
	
	@Test
	public void testJavaTensorToString5() {
		JavaTensor tensor = new JavaTensor(2,2,2);
		String expected = "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]";
		Assert.assertEquals(expected, tensor.toString());
	}
}
