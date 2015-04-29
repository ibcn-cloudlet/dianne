package be.iminds.iot.dianne.tensor;

import java.util.Arrays;
import java.util.Collection;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import be.iminds.iot.dianne.tensor.impl.java.JavaTensorFactory;
import be.iminds.iot.dianne.tensor.impl.nd4j.ND4JTensorFactory;
import be.iminds.iot.dianne.tensor.impl.th.THTensorFactory;

@RunWith(Parameterized.class)
public class TensorTest<T extends Tensor<T>> {

	private TensorFactory<T> factory;

	public TensorTest(TensorFactory<T> f, String name) {
		this.factory = f;
	}

	@Parameters(name="{1}")
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] { 
//				{ new JavaTensorFactory(), "Java Tensor" },
//				{ new ND4JTensorFactory(), "ND4J Tensor" },
				{ new THTensorFactory(), "TH Tensor" }
		});
	}

	@Test
	public void test1DTensor() {
		T t = factory.createTensor(4);
		t.fill(0.0f);
		Assert.assertEquals(1, t.dim());
		Assert.assertEquals(4, t.size());
		Assert.assertEquals(4, t.size(0));

		Assert.assertEquals(0.0f, t.get(1), 0.01);
		t.set(1.0f, 1);
		Assert.assertEquals(1.0f, t.get(1), 0.01);
	}

	@Test
	public void test2DTensor() {
		T t = factory.createTensor(3, 4);
		t.fill(0.0f);
		Assert.assertEquals(2, t.dim());
		Assert.assertEquals(12, t.size());
		Assert.assertEquals(3, t.size(0));
		Assert.assertEquals(4, t.size(1));

		Assert.assertEquals(0.0f, t.get(1, 2), 0.01);
		t.set(1.0f, 1, 2);
		Assert.assertEquals(1.0f, t.get(1, 2), 0.01);

	}

	@Test
	public void test2DTensor2() {
		float[] data = new float[]{1.0f, 2.0f, 3.0f, 4.0f};
		T t = factory.createTensor(data, 2, 2);
		Assert.assertEquals(2, t.dim());
		Assert.assertEquals(4, t.size());
		Assert.assertEquals(2, t.size(0));
		Assert.assertEquals(2, t.size(1));

		Assert.assertEquals(1.0f, t.get(0, 0), 0.01);
		
		float[] newdata = t.get();
		Assert.assertArrayEquals(data, newdata, 0.01f);

	}
	
	@Test
	public void test3DTensor() {
		T t = factory.createTensor(2, 3, 4);
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
		T t = factory.createTensor(2, 2);
		t.fill(1.0f);
		T t2 = factory.createTensor(2, 2);
		t2.fill(1.0f);
		T t3 = factory.createTensor(4);
		t3.fill(1.0f);

		Assert.assertEquals(true, t.equals(t2));
		Assert.assertEquals(false, t.equals(t3));
		t.set(2.0f, 0, 0);
		Assert.assertEquals(false, t.equals(t2));
	}
	
	@Test
	public void testReshape() {
		T t = factory.createTensor(2, 2);
		t.fill(1.0f);
		T t2 = factory.createTensor(4);
		t2.fill(1.0f);

		t.reshape(4);
		Assert.assertEquals(t2, t);
	}

	@Test
	public void testCopyInto() {
		T t = factory.createTensor(2, 2);
		t.set(0f, 0, 0);
		t.set(1f, 0, 1);
		t.set(2f, 1, 0);
		t.set(3f, 1, 1);

		T t2 = t.copyInto(null);
		Assert.assertEquals(0f, t2.get(0, 0), 0.1f);
		Assert.assertEquals(1f, t2.get(0, 1), 0.1f);
		Assert.assertEquals(2f, t2.get(1, 0), 0.1f);
		Assert.assertEquals(3f, t2.get(1, 1), 0.1f);

		T t3 = factory.createTensor(4);
		t.copyInto(t3);
		
		Assert.assertEquals(0f, t3.get(0), 0.1f);
		Assert.assertEquals(1f, t3.get(1), 0.1f);
		Assert.assertEquals(2f, t3.get(2), 0.1f);
		Assert.assertEquals(3f, t3.get(3), 0.1f);
	}

	@Test
	public void testTranspose() {
		T t = factory.createTensor(2, 3);
		for (int i = 0; i < 2; i++)
			for (int j = 0; j < 3; j++)
				t.set(i * 2 + j, i, j);

		T t2 = t.transpose(null, 0, 1);
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 2; j++)
				Assert.assertEquals(j * 2 + i, t2.get(i, j), 0.1f);
	}

	@Test
	public void testNarrow1() {
		T t = factory.createTensor(5, 6);
		t.fill(0.0f);
		T t2 = t.narrow(0, 1, 3);
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
		T t = factory.createTensor(5, 6);
		t.fill(0.0f);
		T t2 = t.narrow(0, 1, 3);
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
		T t = factory.createTensor(5, 6);
		t.fill(0.0f);
		T t2 = t.narrow(1, 3, 2, 3);
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
	public void testCopyIntoNarrow() {
		T t = factory.createTensor(2, 3);
		for (int i = 0; i < 2; i++)
			for (int j = 0; j < 3; j++)
				t.set(i * 2 + j, i, j);

		T t2 = t.narrow(0, 1, 1);
		T t3 = t2.copyInto(null);

		Assert.assertEquals(true, t3.equals(t2));
		Assert.assertEquals(3, t3.size());
	}

	@Test
	public void testSelect() {
		T t = factory.createTensor(3, 3);
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				t.set(i * 2 + j, i, j);

		T row1 = t.select(0, 1);
		T expected = factory.createTensor(3);
		expected.set(2.0f, 0);
		expected.set(3.0f, 1);
		expected.set(4.0f, 2);

		Assert.assertEquals(1, row1.dim());
		Assert.assertEquals(3, row1.dims()[0]);
		Assert.assertEquals(expected, row1);

		T col1 = t.select(1, 1);

		expected.set(1.0f, 0);
		expected.set(3.0f, 1);
		expected.set(5.0f, 2);

		Assert.assertEquals(1, col1.dim());
		Assert.assertEquals(3, col1.dims()[0]);
		Assert.assertEquals(expected, col1);
	}

	@Test
	public void testDiag() {
		T t = factory.createTensor(3, 3);
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				t.set(i * 2 + j, i, j);

		T diag = t.diag(null);
		T expected = factory.createTensor(3);
		expected.set(0.0f, 0);
		expected.set(3.0f, 1);
		expected.set(6.0f, 2);
		Assert.assertEquals(expected, diag);
	}
}
