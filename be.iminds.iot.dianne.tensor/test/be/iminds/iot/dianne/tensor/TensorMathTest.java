package be.iminds.iot.dianne.tensor;

import java.util.Arrays;
import java.util.Collection;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import be.iminds.iot.dianne.tensor.impl.java.JavaTensorFactory;
import be.iminds.iot.dianne.tensor.impl.th.THTensorFactory;

@RunWith(Parameterized.class)
public class TensorMathTest<T extends Tensor<T>> {

	private TensorFactory<T> factory;
	private TensorMath<T> math;

	public TensorMathTest(TensorFactory<T> f, String name) {
		this.factory = f;
	}

	@Parameters(name="{1}")
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] { 
				{ new JavaTensorFactory(), "Java Tensor" },
				{ new THTensorFactory(), "TH Tensor" }
		});
	}
	
    @Before
    public void setUp() {
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
	public void testAdd4() {
		T t1 = factory.createTensor(2,2);
		t1.fill(2);
		T t2 = factory.createTensor(4);
		t2.fill(3);
		
		T r = factory.createTensor(2,2);
		r.fill(1.0f);
		math.add(r, t1, 2, t2);
		
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
	public void testSub3() {
		T t1 = factory.createTensor(2,2);
		t1.fill(3);
		T t2 = factory.createTensor(4);
		t2.fill(1);
		
		T r = factory.createTensor(2,2);
		r.fill(1.0f);
		math.sub(r, t1, t2);
		
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
	public void testMul2() {
		T t1 = factory.createTensor(2,2);
		t1.fill(2);
		
		T r = factory.createTensor(2,2);
		r.fill(1.0f);
		math.mul(r, t1, 2);
		
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
	public void testCMul2() {
		T t1 = factory.createTensor(2,2);
		t1.fill(2);
		T t2 = factory.createTensor(4);
		t2.fill(3);
		
		T r = factory.createTensor(2,2);
		r.fill(1.0f);
		math.cmul(r, t1, t2);
		
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
	public void testDiv2() {
		T t1 = factory.createTensor(2,2);
		t1.fill(6);
		
		T r = factory.createTensor(2,2);
		r.fill(1.0f);
		math.div(r, t1, 2);
		
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
	public void testCDiv2() {
		T t1 = factory.createTensor(2,2);
		t1.fill(6);
		T t2 = factory.createTensor(4);
		t2.fill(3);
		
		T r = factory.createTensor(2,2);
		r.fill(1.0f);
		math.cdiv(r, t1, t2);
		
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
		
		Assert.assertEquals(24f, dot, 0.01);
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
	public void testMv2() {
		T t1 = factory.createTensor(2,2);
		t1.fill(2);
		T t2 = factory.createTensor(2);
		t2.fill(3);
		
		T r = factory.createTensor(2);
		r.fill(1.0f);
		math.mv(r, t1, t2);
		
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
	public void testMm3() {
		T t1 = factory.createTensor(2,2);
		t1.fill(2);
		T t2 = factory.createTensor(2,2);
		t2.fill(3);
		
		T r = factory.createTensor(2,2);
		r.fill(1.0f);
		math.mm(r, t1, t2);
		
		T exp = factory.createTensor(2,2);
		exp.fill(12);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testAddmv() {
		T t1 = factory.createTensor(2,2);
		t1.fill(2);
		T t2 = factory.createTensor(2);
		t2.fill(3);
		T t3 = factory.createTensor(2);
		t3.fill(1.0f);
		
		T r = math.addmv(null, t3, t1, t2);
		
		T exp = factory.createTensor(2);
		exp.fill(13);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testAddmv2() {
		T t1 = factory.createTensor(2,2);
		t1.fill(2);
		T t2 = factory.createTensor(2);
		t2.fill(3);
		T t3 = factory.createTensor(2);
		t3.fill(1.0f);
		
		T r = factory.createTensor(2);
		r.fill(1.0f);
		math.addmv(r, t3, t1, t2);
		
		T exp = factory.createTensor(2);
		exp.fill(13);
		
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
		t1.set(1, 1);
		t1.set(5, 3);

		Assert.assertEquals(1.0, math.min(t1), 0.1f);
	}
	
	@Test
	public void testMean() {
		T t1 = factory.createTensor(4);
		t1.fill(2);
		t1.set(0, 0);
		t1.set(5, 3);
		
		Assert.assertEquals(2.25, math.mean(t1), 0.1f);
	}
	
	@Test
	public void testArgMax() {
		T t1 = factory.createTensor(4);
		t1.fill(2);
		t1.set(0, 0);
		t1.set(5, 3);
		
		Assert.assertEquals(3, math.argmax(t1));
	}
	
	@Test
	public void testArgMin() {
		T t1 = factory.createTensor(4);
		t1.fill(2);
		t1.set(0, 0);
		t1.set(5, 3);
		Assert.assertEquals(0, math.argmin(t1));
	}
	
	@Test
	public void testTanh(){
		float[] data = new float[10];
		for(int i=0;i<data.length;i++){
			data[i] = i;
		}
		T t1 = factory.createTensor(data, 10);
	
		float[] expdata = new float[10];
		for(int i=0;i<expdata.length;i++){
			expdata[i] = (float) Math.tanh((double)i);
		}
		T exp = factory.createTensor(expdata, 10);
		
		Assert.assertEquals(exp, math.tanh(null, t1));
	}
	
	@Test
	public void testDTanh(){
		float[] data = new float[10];
		for(int i=0;i<data.length;i++){
			data[i] = i;
		}
		T t1 = factory.createTensor(data, 10);
	
		float[] expdata = new float[10];
		for(int i=0;i<expdata.length;i++){
			expdata[i] = 1-i*i;
		}
		T exp = factory.createTensor(expdata, 10);
		
		Assert.assertEquals(exp, math.dtanh(null, t1));
	}
	
	@Test
	public void testSigmoid(){
		float[] data = new float[10];
		for(int i=0;i<data.length;i++){
			data[i] = i;
		}
		T t1 = factory.createTensor(data, 10);
	
		float[] expdata = new float[10];
		for(int i=0;i<expdata.length;i++){
			expdata[i] = (float) (1.0f/(1.0f + Math.exp(-i)));
		}
		T exp = factory.createTensor(expdata, 10);
		
		Assert.assertEquals(exp, math.sigmoid(null, t1));
	}
	
	@Test
	public void testDSigmoid(){
		float[] data = new float[10];
		for(int i=0;i<data.length;i++){
			data[i] = i;
		}
		T t1 = factory.createTensor(data, 10);
	
		float[] expdata = new float[10];
		for(int i=0;i<expdata.length;i++){
			expdata[i] = (1.0f - i) * i;
		}
		T exp = factory.createTensor(expdata, 10);
		
		Assert.assertEquals(exp, math.dsigmoid(null, t1));
	}
	
	@Test
	public void testThreshold(){
		float threshold = 5.5f;
		float[] data = new float[10];
		for(int i=0;i<data.length;i++){
			data[i] = i;
		}
		T t1 = factory.createTensor(data, 10);
	
		float[] expdata = new float[10];
		for(int i=0;i<expdata.length;i++){
			expdata[i] = i > threshold ? i : 0;
		}
		T exp = factory.createTensor(expdata, 10);
		
		Assert.assertEquals(exp, math.thresh(null, t1, threshold, 0, 0));
	}
	
	@Test
	public void testDThreshold(){
		float threshold = 5.5f;
		float[] data = new float[10];
		for(int i=0;i<data.length;i++){
			data[i] = i;
		}
		T t1 = factory.createTensor(data, 10);
	
		float[] expdata = new float[10];
		for(int i=0;i<expdata.length;i++){
			expdata[i] = i > threshold ? 1 : 0;
		}
		T exp = factory.createTensor(expdata, 10);
		
		Assert.assertEquals(exp, math.dthresh(null, t1, threshold, 0));
	}
	
	@Test
	public void testSoftmax(){
		float[] data = new float[10];
		for(int i=0;i<data.length;i++){
			data[i] = i;
		}
		T t1 = factory.createTensor(data, 10);
	
		float[] expdata = new float[]{7.8013414E-5f, 2.1206246E-4f, 5.764455E-4f, 0.0015669414f, 0.004259388f, 0.011578218f, 0.031472858f, 0.0855521f, 0.23255472f, 0.6321493f};
		T exp = factory.createTensor(expdata, 10);
		
		Assert.assertTrue(exp.equals(math.softmax(null, t1), 0.000001f));
	}
	
	@Test
	public void testConvolution(){
		T t1 = factory.createTensor(5,5);
		T t2 = factory.createTensor(3,3);
		T exp = factory.createTensor(3,3);
		
		t1.fill(1.0f);
		t1.set(2.0f, 0, 0);
		t1.set(2.0f, 4, 4);
		t2.fill(2.0f);
		exp.fill(18.0f);
		exp.set(20.0f, 0, 0);
		exp.set(20.0f, 2, 2);
		
		Assert.assertEquals(exp, math.convolution2D(null, t1, t2, 1, 1, 0, false));
	}
	
	@Test
	public void testConvolution2(){
		T t1 = factory.createTensor(5,5);
		T t2 = factory.createTensor(3,3);
		T exp = factory.createTensor(3,3);
		
		t1.fill(1.0f);
		t1.set(2.0f, 0, 0);
		t1.set(2.0f, 4, 4);
		t2.fill(2.0f);
		exp.fill(18.0f);
		exp.set(20.0f, 0, 0);
		exp.set(20.0f, 2, 2);
		
		T r = factory.createTensor(3,3);
		r.fill(1.0f);
		math.convolution2D(r, t1, t2, 1, 1, 0, false);

		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testConvolution3(){
		float[] data = new float[25];
		for(int i=0;i<data.length;i++){
			data[i] = i;
		}
		T t1 = factory.createTensor(data, 5,5);
		T t2 = factory.createTensor(3,3);
		t2.fill(2.0f);
		
		float[] e = new float[]{
				108, 126, 144,
				198, 216, 234,
				288, 306, 324
		};
		T exp = factory.createTensor(e, 3,3);
		
		Assert.assertEquals(exp, math.convolution2D(null, t1, t2, 1, 1, 0, false));
	}
	
	@Test
	public void testConvolutionStride(){
		float[] data = new float[25];
		for(int i=0;i<data.length;i++){
			data[i] = i;
		}
		T t1 = factory.createTensor(data, 5,5);
		T t2 = factory.createTensor(3,3);
		t2.fill(2.0f);
		
		float[] e = new float[]{
				108, 144,
				288, 324
		};
		T exp = factory.createTensor(e, 2,2);
		
		Assert.assertEquals(exp, math.convolution2D(null, t1, t2, 2, 2, 0, false));
	}
	@Test
	public void testConvolutionFull(){
		T t1 = factory.createTensor(5,5);
		T t2 = factory.createTensor(3,3);
		t1.fill(1.0f);
		t1.set(2.0f, 0, 0);
		t1.set(2.0f, 4, 4);
		t2.fill(2.0f);
		
		float[] data = new float[]{
			4.0f, 6.0f,  8.0f,  6.0f,  6.0f,  4.0f,  2.0f,
			6.0f, 10.0f, 14.0f, 12.0f, 12.0f, 8.0f,  4.0f,
			8.0f, 14.0f, 20.0f, 18.0f, 18.0f, 12.0f, 6.0f,
			6.0f, 12.0f, 18.0f, 18.0f, 18.0f, 12.0f, 6.0f,
			6.0f, 12.0f, 18.0f, 18.0f, 20.0f, 14.0f, 8.0f,
			4.0f, 8.0f,  12.0f, 12.0f, 14.0f, 10.0f, 6.0f,
			2.0f, 4.0f,  6.0f,  6.0f,  8.0f,  6.0f,  4.0f
		};
		T exp = factory.createTensor(data, 7,7);
		
		Assert.assertEquals(exp, math.convolution2D(null, t1, t2, 1, 1, 1, false));
	}
	
	@Test
	public void testConvolutionSame(){
		T t1 = factory.createTensor(5,5);
		T t2 = factory.createTensor(3,3);
		t1.fill(1.0f);
		t1.set(2.0f, 0, 0);
		t1.set(2.0f, 4, 4);
		t2.fill(2.0f);
		
		// add zero padding for same convolution
		T padded = math.zeropad(null, t1, 1, 1);
		
		float[] data = new float[]{
				10.0f, 14.0f, 12.0f, 12.0f, 8.0f,
				14.0f, 20.0f, 18.0f, 18.0f, 12.0f,
				12.0f, 18.0f, 18.0f, 18.0f, 12.0f,
				12.0f, 18.0f, 18.0f, 20.0f, 14.0f,
				8.0f,  12.0f, 12.0f, 14.0f, 10.0f,
		};
		T exp = factory.createTensor(data, 5, 5);
	
		Assert.assertEquals(exp, math.convolution2D(null, padded, t2, 1, 1, 0, false));
	}
	
	@Test
	public void testConvolutionAdd(){
		float[] data = new float[25];
		for(int i=0;i<data.length;i++){
			data[i] = i;
		}
		T t1 = factory.createTensor(data, 5,5);
		T t2 = factory.createTensor(3,3);
		t2.fill(2.0f);
		T t3 = factory.createTensor(3,3);
		t3.fill(2.0f);
		
		float[] e = new float[]{
				110, 128, 146,
				200, 218, 236,
				290, 308, 326
		};
		T exp = factory.createTensor(e, 3,3);
		
		Assert.assertEquals(exp, math.addconvolution2D(null, t3, t1, t2, 1, 1, 0, false));
	}
	
	@Test
	public void testMaxpool(){
		T t1 = factory.createTensor(4,6);
		t1.fill(0.0f);
		t1.set(1.0f, 0, 0);
		t1.set(1.0f, 0, 3);
		t1.set(2.0f, 2, 0);
		t1.set(4.0f, 0, 5);
		
		T exp = factory.createTensor(2,3);
		exp.fill(0.0f);
		exp.set(1.0f, 0, 0);
		exp.set(1.0f, 0, 1);
		exp.set(2.0f, 1, 0);
		exp.set(0.0f, 1, 1);
		exp.set(4.0f, 0, 2);
		
		Assert.assertEquals(exp, math.maxpool2D(null, t1, 2, 2, 2, 2));
	}

	@Test
	public void testMaxpool2(){
		T t1 = factory.createTensor(4,6);
		t1.fill(0.0f);
		t1.set(1.0f, 0, 0);
		t1.set(1.0f, 0, 3);
		t1.set(2.0f, 2, 0);
		t1.set(4.0f, 0, 5);
		
		T exp = factory.createTensor(2,3);
		exp.fill(0.0f);
		exp.set(1.0f, 0, 0);
		exp.set(1.0f, 0, 1);
		exp.set(2.0f, 1, 0);
		exp.set(0.0f, 1, 1);
		exp.set(4.0f, 0, 2);
		
		T r = factory.createTensor(2,3);
		r.fill(1.0f);
		math.maxpool2D(r, t1, 2, 2, 2, 2);
		
		Assert.assertEquals(exp, r);
	}
	
	@Test
	public void testMaxpoolStride(){
		T t1 = factory.createTensor(4,6);
		t1.fill(0.0f);
		t1.set(1.0f, 0, 0);
		t1.set(1.0f, 0, 3);
		t1.set(2.0f, 2, 0);
		t1.set(4.0f, 0, 5);
		
		T exp = factory.createTensor(3,5);
		exp.fill(0.0f);
		exp.set(1.0f, 0, 0);
		exp.set(1.0f, 0, 2);
		exp.set(1.0f, 0, 3);
		exp.set(4.0f, 0, 4);
		exp.set(2.0f, 1, 0);
		exp.set(2.0f, 2, 0);

		Assert.assertEquals(exp, math.maxpool2D(null, t1, 2, 2, 1, 1));
	}
	
	
	@Test
	public void testDMaxpool(){
		T t1 = factory.createTensor(4,6);
		t1.fill(0.0f);
		t1.set(1.0f, 0, 0);
		t1.set(1.0f, 0, 3);
		t1.set(2.0f, 2, 0);
		t1.set(4.0f, 0, 5);
		
		T grad = factory.createTensor(2,3);
		grad.set(1.0f, 0, 0);
		grad.set(1.0f, 0, 1);
		grad.set(1.0f, 1, 0);
		grad.set(0.0f, 1, 1);
		grad.set(1.0f, 0, 2);
		
		T exp = factory.createTensor(4,6);
		exp.fill(0.0f);
		exp.set(1.0f, 0, 0);
		exp.set(1.0f, 0, 3);
		exp.set(1.0f, 2, 0);
		exp.set(1.0f, 0, 5);
	
		Assert.assertEquals(exp, math.dmaxpool2D(null, grad, t1, 2, 2, 2 ,2));
	}
	
	@Test
	public void testZeropad(){
		T t1 = factory.createTensor(3,3);
		t1.fill(2.0f);
		
		float[] data = new float[]{
				0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
				0.0f, 2.0f, 2.0f, 2.0f, 0.0f,
				0.0f, 2.0f, 2.0f, 2.0f, 0.0f,
				0.0f, 2.0f, 2.0f, 2.0f, 0.0f,
				0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		};
		T exp = factory.createTensor(data, 5, 5);
	
		Assert.assertEquals(exp, math.zeropad(null, t1, 1, 1));
	}
	
	@Test
	public void testSpatialConvolution(){
		T t = factory.createTensor(3,5,5);
		t.fill(1.0f);
		
		T add = factory.createTensor(5);
		add.fill(1.0f);
		T k = factory.createTensor(5, 3, 3, 3);
		k.fill(1.0f);
		
		T exp = factory.createTensor(5, 3, 3);
		exp.fill(28.0f);
		
		Assert.assertEquals(exp, math.spatialconvolve(null, add, t, k, 1, 1, 0, 0));
	}
	
	@Test
	public void testSpatialConvolutionPadded(){
		T t = factory.createTensor(3,5,5);
		t.fill(1.0f);
		
		T add = factory.createTensor(5);
		add.fill(1.0f);
		T k = factory.createTensor(5, 3, 3, 3);
		k.fill(1.0f);
		
		float[] data = new float[]{13.0f, 19.0f, 19.0f, 19.0f, 13.0f, 19.0f, 28.0f, 28.0f, 28.0f, 19.0f, 19.0f, 28.0f, 28.0f, 28.0f, 19.0f, 19.0f, 28.0f, 28.0f, 28.0f, 19.0f, 13.0f, 19.0f, 19.0f, 19.0f, 13.0f, 13.0f, 19.0f, 19.0f, 19.0f, 13.0f, 19.0f, 28.0f, 28.0f, 28.0f, 19.0f, 19.0f, 28.0f, 28.0f, 28.0f, 19.0f, 19.0f, 28.0f, 28.0f, 28.0f, 19.0f, 13.0f, 19.0f, 19.0f, 19.0f, 13.0f, 13.0f, 19.0f, 19.0f, 19.0f, 13.0f, 19.0f, 28.0f, 28.0f, 28.0f, 19.0f, 19.0f, 28.0f, 28.0f, 28.0f, 19.0f, 19.0f, 28.0f, 28.0f, 28.0f, 19.0f, 13.0f, 19.0f, 19.0f, 19.0f, 13.0f, 13.0f, 19.0f, 19.0f, 19.0f, 13.0f, 19.0f, 28.0f, 28.0f, 28.0f, 19.0f, 19.0f, 28.0f, 28.0f, 28.0f, 19.0f, 19.0f, 28.0f, 28.0f, 28.0f, 19.0f, 13.0f, 19.0f, 19.0f, 19.0f, 13.0f, 13.0f, 19.0f, 19.0f, 19.0f, 13.0f, 19.0f, 28.0f, 28.0f, 28.0f, 19.0f, 19.0f, 28.0f, 28.0f, 28.0f, 19.0f, 19.0f, 28.0f, 28.0f, 28.0f, 19.0f, 13.0f, 19.0f, 19.0f, 19.0f, 13.0f};
		T exp = factory.createTensor(data, 5, 5, 5);
		
		Assert.assertEquals(exp, math.spatialconvolve(null, add, t, k, 1, 1, 1, 1));
	}
	
	@Test
	public void testSpatialMaxpool(){
		T t = factory.createTensor(3,5,5);
		t.fill(1.0f);
		
	
		T exp = factory.createTensor(3, 2, 2);
		exp.fill(1.0f);

		Assert.assertEquals(exp, math.spatialmaxpool(null, t, 2, 2, 2, 2));
	}
	
	@Test
	public void testScaleUp(){
		float[] data = new float[]{0, 0, 0, 0, 1, 0, 0, 0, 0}; 
		T t = factory.createTensor(data, 3, 3);
		
		float[] expData = new float[]{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.16000001f, 0.32000002f, 0.32f, 0.16f, 0.0f, 0.0f, 0.32000002f, 0.64000005f, 0.64f, 0.32f, 0.0f, 0.0f, 0.32f, 0.64f, 0.6399999f, 0.31999996f, 0.0f, 0.0f, 0.16f, 0.32f, 0.31999996f, 0.15999998f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
		T exp = factory.createTensor(expData, 6, 6);
		
		T result = factory.getTensorMath().scale2D(null, t, new int[]{6, 6});
		Assert.assertEquals(exp, result);
	}
	
	@Test
	public void testScaleUp2(){
		float[] data = new float[]{0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0}; 
		T t = factory.createTensor(data, 2, 3, 3);
		
		float[] expData = new float[]{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.16000001f, 0.32000002f, 0.32f, 0.16f, 0.0f, 0.0f, 0.32000002f, 0.64000005f, 0.64f, 0.32f, 0.0f, 0.0f, 0.32f, 0.64f, 0.6399999f, 0.31999996f, 0.0f, 0.0f, 0.16f, 0.32f, 0.31999996f, 0.15999998f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
									  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.32000002f, 0.64000005f, 0.64f, 0.32f, 0.0f, 0.0f, 0.64000005f, 1.2800001f, 1.28f, 0.64f, 0.0f, 0.0f, 0.64f, 1.28f, 1.2799999f, 0.6399999f, 0.0f, 0.0f, 0.32f, 0.64f, 0.6399999f, 0.31999996f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

		T exp = factory.createTensor(expData, 2, 6, 6);
		
		T result = factory.getTensorMath().scale2D(null, t, new int[]{2, 6, 6});
		Assert.assertEquals(exp, result);
	}
	
	@Test
	public void testScaleUp3(){
		T t = factory.createTensor(2, 4, 4);
		t.fill(1.0f);
		
		T exp = factory.createTensor(2, 15, 15);
		exp.fill(1.0f);
		
		T result = factory.getTensorMath().scale2D(null, t, new int[]{2, 15, 15});
		Assert.assertTrue(exp.equals(result, 0.00001f));
	}
	
	@Test
	public void testScaleDown(){
		float[] data = new float[]{0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0}; 
		T t = factory.createTensor(data, 4, 4);
		
		float[] expData = new float[]{0, 0, 0, 0, 1, 0, 0, 0, 0};
		T exp = factory.createTensor(expData, 3, 3);
		
		T result = factory.getTensorMath().scale2D(null, t, new int[]{3, 3});
		Assert.assertEquals(exp, result);
	}
	
	@Test
	public void testScaleDown2(){
		float[] data = new float[]{0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
				                   0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0}; 
		T t = factory.createTensor(data, 2, 4, 4);
		
		float[] expData = new float[]{0, 0, 0, 0, 1, 0, 0, 0, 0,
				                      0, 0, 0, 0, 2, 0, 0, 0, 0};
		T exp = factory.createTensor(expData, 2, 3, 3);
		
		T result = factory.getTensorMath().scale2D(null, t, new int[]{2, 3, 3});
		Assert.assertEquals(exp, result);
	}
	
	@Test
	public void testScaleDown3(){
		T t = factory.createTensor(16, 16);
		t.fill(1.0f);
		
		T exp = factory.createTensor(3, 4, 4);
		exp.fill(1.0f);
		
		T result = factory.getTensorMath().scale2D(null, t, new int[]{3, 4, 4});
		Assert.assertTrue(exp.equals(result, 0.00001f));
	}
}
