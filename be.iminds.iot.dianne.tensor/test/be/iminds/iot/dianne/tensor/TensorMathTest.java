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
package be.iminds.iot.dianne.tensor;

import java.util.Arrays;
import java.util.Collection;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Ignore;
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
	public void testSpatialConvolution1(){
		// given tensors created with Torch7
		float[] input = new float[]{0.77609068155289f, 0.55822139978409f, 0.91193771362305f, 0.78901469707489f, 0.42735171318054f, 0.79527574777603f, 0.67515754699707f, 0.53678107261658f, 0.15673413872719f, 0.022478198632598f, 0.72448128461838f, 0.96998769044876f, 0.019700326025486f, 0.27237677574158f, 0.030741902068257f, 0.96200960874557f, 0.065508991479874f, 0.45543125271797f, 0.047680865973234f, 0.26728358864784f, 0.72873586416245f, 0.95224910974503f, 0.95607620477676f, 0.33243060112f, 0.52823734283447f, 0.81236803531647f, 0.60130661725998f, 0.18064060807228f, 0.072066083550453f, 0.57467979192734f, 0.95971059799194f, 0.56326526403427f, 0.085398368537426f, 0.0576012134552f, 0.50791573524475f, 0.14721287786961f, 0.46993172168732f, 0.44745245575905f, 0.093035534024239f, 0.19360491633415f, 0.60368782281876f, 0.17873245477676f, 0.88706678152084f, 0.13631945848465f, 0.10627333074808f, 0.95762068033218f, 0.14315581321716f, 0.63736015558243f, 0.51555347442627f, 0.23638533055782f, 0.51193690299988f, 0.35108935832977f, 0.53834712505341f, 0.001205736072734f, 0.21984222531319f, 0.57446950674057f, 0.059494987130165f, 0.94047862291336f, 0.63164830207825f, 0.85614901781082f, 0.96907782554626f, 0.79864066839218f, 0.70667326450348f, 0.069181926548481f, 0.38727232813835f, 0.71341925859451f, 0.57678472995758f, 0.28965455293655f, 0.56956791877747f, 0.54622274637222f, 0.12288309633732f, 0.061499424278736f,};
		T t = factory.createTensor(input, 2,6,6);
		float[] kernel = new float[]{-0.14343239367008f, -0.093330025672913f, 0.11067543923855f, 0.015021086670458f, 0.024092333391309f, -0.17889711260796f, 0.15343734622002f, 0.076087519526482f, 0.073676384985447f, -0.078303165733814f, 0.22008927166462f, 0.0094112325459719f, -0.076281048357487f, 0.15629014372826f, -0.010006110183895f, -0.057917769998312f, -0.21304705739021f, -0.092932909727097f, -0.19206054508686f, -0.0923076197505f, -0.14253155887127f, 0.023537524044514f, 0.15369735658169f, -0.040302261710167f, 0.15848252177238f, 0.035916920751333f, -0.091992326080799f, 0.084421306848526f, -0.11242455989122f, -0.13386431336403f, 0.12271160632372f, -0.23530527949333f, -0.068826608359814f, -0.20107996463776f, 0.22253783047199f, 0.16683535277843f, };
		T k = factory.createTensor(kernel, 2, 2, 3, 3);
		T b = factory.createTensor(2); // 0 bias
		b.fill(0.0f);
		float[] output = new float[]{-0.15410257875919f, -0.12652185559273f, -0.17445874214172f, -0.10743043571711f, -0.23350170254707f, -0.14293836057186f, 0.27394169569016f, -0.034324631094933f, -0.16394764184952f, -0.050246603786945f, -0.026055742055178f, -0.083681374788284f, 0.17076663672924f, -0.072506308555603f, -0.031656593084335f, -0.17103688418865f, -0.14768862724304f, -0.26008188724518f, -0.5494099855423f, -0.10471825301647f, -0.099049426615238f, -0.35708567500114f, 0.20025815069675f, -0.050372511148453f, -0.026007436215878f, -0.063242539763451f, -0.27179110050201f, -0.053310103714466f, 0.011701263487339f, -0.060526646673679f, -0.470194876194f, -0.62272024154663f, };
		T exp = factory.createTensor(output, 2, 4, 4);
		
		T res = math.spatialconvolve(null, b, t, k, 1, 1, 0, 0);
		Assert.assertEquals(true, res.sameDim(exp));
		float[] d = res.get();
		for(int i=0;i<output.length;i++){
			Assert.assertEquals(output[i], d[i], 0.001f);
		}
	}
	
	@Test
	public void testSpatialConvolution2(){
		// given tensors created with Torch7
		float[] input = new float[]{0.31436881422997f, 0.24950990080833f, 0.66748958826065f, 0.30248141288757f, 0.27733883261681f, 0.77093708515167f, 0.80574238300323f, 0.018291387706995f, 0.41377124190331f, 0.44120571017265f, 0.18945832550526f, 0.59040051698685f, 0.99503117799759f, 0.45531353354454f, 0.50029301643372f, 0.62687504291534f, 0.58782958984375f, 0.13957872986794f, 0.53331089019775f, 0.7109871506691f, 0.33000388741493f, 0.51233291625977f, 0.63732051849365f, 0.68862748146057f, 0.1111683472991f,};
		T t = factory.createTensor(input, 1,5,5);
		float[] kernel = new float[]{0.021569160744548f, 0.052451651543379f, 0.099111422896385f, -0.17179603874683f, 0.014340420253575f, 0.16796870529652f, -0.082419194281101f, -0.17123277485371f, 0.045430954545736f,};
		T k = factory.createTensor(kernel, 1, 1, 3, 3);
		T b = factory.createTensor(1); // 0 bias
		b.fill(0.0f);
		float[] output = new float[]{-0.10329885780811f, -0.19664537906647f, -0.0025951750576496f, 0.05777532607317f, 0.00054709985852242f, -0.085076838731766f, -0.028119929134846f, -0.019424609839916f, 0.032628562301397f,};
		T exp = factory.createTensor(output, 1, 3, 3);
		T res = factory.createTensor(1,3,3);
		
		T res2 = math.spatialconvolve(res, b, t, k, 1, 1, 0, 0);
		Assert.assertEquals(res, res2);
		Assert.assertEquals(true, res.sameDim(exp));
		float[] d = res.get();
		for(int i=0;i<output.length;i++){
			Assert.assertEquals(output[i], d[i], 0.001f);
		}
	}
	
	@Test
	public void testSpatialConvolution3(){
		// given tensors created with Torch7
		float[] input = new float[]{0.19151945412159f, 0.49766367673874f, 0.62210875749588f, 0.81783843040466f, 0.43772774934769f, 0.61211186647415f, 0.78535860776901f, 0.77135992050171f, 0.77997583150864f, 0.86066979169846f, 0.27259260416031f, 0.15063697099686f, 0.27646425366402f, 0.19851875305176f, 0.80187219381332f, 0.81516295671463f, 0.95813935995102f, 0.15881535410881f, 0.8759326338768f, 0.11613783240318f, 0.35781726241112f, 0.012907532975078f, 0.50099509954453f, 0.48683345317841f, 0.68346291780472f, 0.33101543784142f, 0.71270203590393f, 0.80263954401016f, 0.37025076150894f, 0.098251938819885f, 0.56119620800018f, 0.0559934489429f, 0.5030831694603f, 0.44266265630722f, 0.013768452219665f, 0.022143909707665f, 0.7728266119957f, 0.29072853922844f, 0.88264119625092f, 0.24639444053173f, 0.36488598585129f, 0.73828679323196f, 0.61539614200592f, 0.88922613859177f, 0.075381234288216f, 0.98713928461075f, 0.36882400512695f, 0.11744338274002f, 0.93314009904861f, 0.39378234744072f, 0.65137815475464f, 0.45272982120514f, 0.39720258116722f, 0.53814780712128f, 0.78873014450073f, 0.79062211513519f, 0.31683611869812f, 0.46583634614944f, 0.56809866428375f, 0.43533223867416f, 0.86912739276886f, 0.56947863101959f, 0.43617340922356f, 0.96925902366638f, 0.80214762687683f, 0.040556151419878f, 0.14376682043076f, 0.54811954498291f, 0.70426094532013f, 0.462576597929f, 0.7045813202858f, 0.37647223472595f, 0.21879211068153f, 0.32791209220886f, 0.92486763000488f, 0.81352895498276f, 0.44214075803757f, 0.64655232429504f, 0.90931594371796f, 0.047426480799913f, 0.059809219092131f, 0.99495756626129f, 0.18428708612919f, 0.68923562765121f, 0.047355279326439f, 0.92954593896866f, 0.67488092184067f, 0.91811656951904f, 0.59462475776672f, 0.97530174255371f, 0.53331017494202f, 0.39700198173523f, 0.043324064463377f, 0.26262608170509f, 0.56143307685852f, 0.43015137314796f, 0.32966843247414f, 0.76453077793121f, 0.5029668211937f, 0.59973078966141f, 0.1118943169713f, 0.080946952104568f, 0.6071937084198f, 0.70454448461533f, 0.56594467163086f, 0.16401332616806f, 0.0067640654742718f, 0.032349344342947f, 0.61744171380997f, 0.32815036177635f, 0.91212290525436f, 0.47385999560356f, 0.79052412509918f, 0.068084716796875f, 0.99208146333694f, 0.38271069526672f, 0.95880174636841f, 0.11855413764715f, 0.79196411371231f, 0.89632850885391f, 0.28525096178055f, 0.76430851221085f, 0.62491673231125f, 0.37540492415428f, 0.47809380292892f, 0.025814190506935f, 0.1956751793623f, 0.90653091669083f, 0.38231745362282f, 0.78641778230667f, 0.053873687982559f, 0.61931520700455f, 0.45164841413498f, 0.21158893406391f, 0.98200476169586f, 0.29230400919914f, 0.12394270300865f, 0.6615446805954f, 0.11938089877367f, 0.88626098632812f, 0.73852306604385f, 0.67117643356323f, 0.58730363845825f, 0.32968080043793f, 0.47163254022598f, 0.20736180245876f, 0.1071268171072f, 0.011632074601948f, 0.22921857237816f, 0.071543417870998f, 0.89996516704559f, 0.73005592823029f, 0.41675353050232f, 0.9798396229744f, 0.53585165739059f, 0.69001621007919f, 0.0062085120007396f, 0.99057483673096f, 0.30064171552658f, 0.69053429365158f, 0.43689316511154f, 0.94812321662903f, 0.61214900016785f, 0.58364462852478f, 0.91819804906845f, 0.53155326843262f, 0.62573665380478f, 0.16865733265877f, 0.70599758625031f, 0.15880773961544f, 0.14983370900154f, 0.93796533346176f, 0.74606341123581f, 0.71826457977295f, 0.83100700378418f, 0.47654265165329f, 0.63372576236725f, 0.88365334272385f, 0.43830987811089f, 0.40420231223106f, 0.15257278084755f, 0.1714680492878f, 0.56840962171555f, 0.13183201849461f, 0.52822428941727f, 0.41190814971924f, 0.95142877101898f, 0.024856043979526f, 0.48035916686058f, 0.56356078386307f, 0.50255954265594f, 0.78187793493271f, 0.53687816858292f, 0.26705995202065f, 0.81920206546783f, 0.21425496041775f, 0.057115640491247f, 0.17755703628063f, 0.66942173242569f, 0.42926567792892f, 0.7671166062355f, 0.97211313247681f, 0.70811533927917f, 0.046531576663256f, 0.79686719179153f, 0.91735446453094f, 0.55776083469391f, 0.15893007814884f, 0.96583652496338f, 0.94338268041611f, 0.147156894207f, 0.76316154003143f, 0.029647007584572f, 0.053878273814917f, 0.59389346837997f, 0.25408163666725f, 0.11406569927931f, 0.92797261476517f, 0.95080983638763f, 0.83831149339676f, 0.32570740580559f, 0.15692453086376f, 0.1936186850071f, 0.6907759308815f, 0.45781165361404f, 0.36694642901421f, 0.92040258646011f, 0.93747270107269f, 0.87906914949417f, 0.61336481571198f, 0.25261574983597f, 0.69934982061386f, 0.34800878167152f, 0.50294625759125f, 0.18258872628212f, 0.71111088991165f, 0.90179604291916f, 0.13438585400581f, 0.7065281867981f, 0.82893192768097f, 0.72665846347809f, 0.74284589290619f, 0.90008783340454f, 0.45703375339508f, 0.77916377782822f, 0.079103350639343f, 0.5991547703743f, 0.3730466067791f, 0.29112523794174f, 0.93363636732101f, 0.15139526128769f, 0.41872480511665f, 0.33517464995384f, 0.23421156406403f, 0.65755176544189f, 0.57248485088348f, 0.07334253937006f, 0.57211089134216f, 0.055006392300129f, 0.41689297556877f, 0.3231948018074f, 0.62588310241699f, 0.59048181772232f, 0.22036227583885f, 0.85389858484268f, 0.62205910682678f, 0.28706243634224f, 0.47767162322998f, 0.17306722700596f, 0.97434210777283f, 0.13402120769024f, 0.77298468351364f, 0.99465382099152f, 0.027138542383909f, 0.17949786782265f, 0.22102235257626f, 0.3175468146801f, 0.12032831460238f, 0.56829142570496f, 0.17527398467064f, 0.009348576888442f, 0.42946156859398f, 0.90064859390259f, 0.65776920318604f, 0.97724145650864f, 0.56589859724045f, 0.55689465999603f, 0.56903499364853f, 0.084773845970631f, 0.65419602394104f, 0.33300247788429f, 0.3685576915741f, 0.72842866182327f, 0.95238465070724f, 0.14243537187576f, 0.19677047431469f, 0.55246895551682f, 0.84993004798889f, 0.27304324507713f, 0.96045827865601f,};
		T t = factory.createTensor(input, 3,10,10);
		float[] kernel = new float[]{0.12862013280392f, -0.13920547068119f, 0.13828659057617f, -0.13413745164871f, 0.11005226522684f, 0.12815648317337f, 0.036858595907688f, -0.027259828522801f, 0.12427075207233f, -0.01503942720592f, 0.064891271293163f, 0.081589959561825f, 0.062549516558647f, 0.021417614072561f, -0.13247920572758f, 0.10866478830576f, -0.017470106482506f, -0.040303289890289f, -0.062911033630371f, -0.020988393574953f, -0.047632560133934f, 0.096369333565235f, -0.12022539228201f, -0.11196570843458f, 0.07530065625906f, 0.10645768791437f, 0.0026767237577587f, 0.08410432934761f, 0.046490382403135f, -0.0024036066606641f, 0.037618536502123f, -0.0300721693784f, -0.037260212004185f, -0.13406148552895f, -0.015374792739749f, -0.094775147736073f, -0.024506144225597f, -0.089765071868896f, -0.0056613492779434f, -0.11464177817106f, 0.13952349126339f, 0.11808002740145f, -0.036539427936077f, 0.023766241967678f, -0.14075598120689f, 0.11513265967369f, 0.12179300189018f, -0.018763253465295f, 0.10775551944971f, -0.12171473354101f, -0.04279700666666f, -0.079470090568066f, 0.037566192448139f, 0.057049725204706f, -0.041041821241379f, -2.839388571374e-06f, -0.082901738584042f, 0.001235134084709f, -0.079870857298374f, 0.071085512638092f, -0.023219933733344f, 0.10888158529997f, -0.12329019606113f, -0.080518469214439f, 0.043543487787247f, -0.061378251761198f, 0.073810234665871f, 0.14239317178726f, 0.12473687529564f, 0.10438067466021f, -0.035679042339325f, -0.11291666328907f, -0.058622043579817f, -0.038737781345844f, -0.036966491490602f, 0.01890442892909f, 0.094598658382893f, -0.027227668091655f, 0.11579900979996f, -0.13502188026905f, -0.020985588431358f, -0.095443516969681f, -0.14370553195477f, -0.020007334649563f, -0.13222324848175f, 0.080954432487488f, -0.10340707749128f, -0.021023709326982f, 0.02835863083601f, 0.083695843815804f, -0.11360505968332f, -0.032872602343559f, -0.058555155992508f, 0.13830436766148f, -0.12748125195503f, -0.081000685691833f, };
		T k = factory.createTensor(kernel, 2, 3, 4, 4);
		T b = factory.createTensor(2); // 0 bias
		b.fill(0.0f);
		float[] output = new float[]{0.0076760994270444f, -0.13578976690769f, -0.0080136684700847f, 0.32897824048996f, 0.24951237440109f, 0.21872764825821f, -0.055962421000004f, 0.23956504464149f, 0.0080994302406907f, 0.3219102025032f, 0.035134475678205f, -0.024483159184456f, 0.095619037747383f, 0.24004997313023f, 0.40563109517097f, 0.25508019328117f, 0.15520167350769f, 0.25235113501549f, -0.11665915697813f, -0.055031388998032f, 0.0016182083636522f, 0.44153144955635f, 0.20098933577538f, 0.21445429325104f, 0.33527475595474f, 0.58433747291565f, -0.098766572773457f, 0.23564772307873f, 0.3244189620018f, 0.34847268462181f, -0.12429462373257f, 0.47989955544472f, 0.42913740873337f, 0.24410140514374f, 0.18745447695255f, 0.43541431427002f, 0.21985426545143f, 0.39968737959862f, 0.24839504063129f, 0.11427775770426f, 0.3174142241478f, 0.069982804358006f, -0.0005934052169323f, 0.20103193819523f, 0.12074348330498f, 0.2491837143898f, 0.36198884248734f, 0.37991327047348f, 0.37507671117783f, -0.24065744876862f, -0.45297127962112f, -0.2827091217041f, -0.54096585512161f, -0.53778213262558f, -0.22710636258125f, -0.80977743864059f, -0.21721260249615f, -0.17343041300774f, -0.067298643290997f, -0.40803179144859f, -0.39415597915649f, -0.3421268761158f, -0.48225224018097f, -0.23399023711681f, -0.50796163082123f, -0.46713602542877f, -0.2514688372612f, -0.35465198755264f, -0.34168547391891f, -0.62699294090271f, -0.35059675574303f, -0.32972207665443f, -0.26922312378883f, -0.26192325353622f, -0.23035138845444f, -0.58214282989502f, -0.22916549444199f, -0.4243235886097f, -0.26972436904907f, -0.68697088956833f, -0.14005056023598f, -0.36422204971313f, -0.33060532808304f, -0.28743612766266f, -0.10017727315426f, -0.23023122549057f, -0.44775840640068f, -0.50004231929779f, -0.13968712091446f, -0.37801611423492f, -0.3239284157753f, -0.32185304164886f, -0.31374704837799f, -0.18497055768967f, -0.19066563248634f, -0.18038457632065f, -0.45193347334862f, -0.19752904772758f, };
		T exp = factory.createTensor(output, 2, 7, 7);
		
		T res = math.spatialconvolve(null, b, t, k, 1, 1, 0, 0);
		Assert.assertEquals(true, res.sameDim(exp));
		float[] d = res.get();
		for(int i=0;i<output.length;i++){
			Assert.assertEquals(output[i], d[i], 0.001f);
		}
	}
	
	@Test
	public void testSpatialConvolutionStride(){
		// given tensors created with Torch7
		float[] input = new float[]{0.19151945412159f, 0.49766367673874f, 0.62210875749588f, 0.81783843040466f, 0.43772774934769f, 0.61211186647415f, 0.78535860776901f, 0.77135992050171f, 0.77997583150864f, 0.86066979169846f, 0.27259260416031f, 0.15063697099686f, 0.27646425366402f, 0.19851875305176f, 0.80187219381332f, 0.81516295671463f, 0.95813935995102f, 0.15881535410881f, 0.8759326338768f, 0.11613783240318f, 0.35781726241112f, 0.012907532975078f, 0.50099509954453f, 0.48683345317841f, 0.68346291780472f, };
		T t = factory.createTensor(input, 1,5,5);
		float[] kernel = new float[]{-0.26783204078674f, 0.040797460824251f, -0.29600438475609f, 0.0020554438233376f, -0.038224898278713f, -0.32415437698364f, -0.31857073307037f, 0.18188442289829f, -0.13951431214809f, };
		T k = factory.createTensor(kernel, 1, 1, 3, 3);
		T b = factory.createTensor(1); // 0 bias
		b.fill(0.0f);
		float[] output = new float[]{-0.59195256233215f, -0.73388183116913f, -0.41666641831398f, -0.54051500558853f, };
		T exp = factory.createTensor(output, 1, 2, 2);
		
		T res = math.spatialconvolve(null, b, t, k, 2, 2, 0, 0);
		Assert.assertEquals(true, res.sameDim(exp));
		float[] d = res.get();
		for(int i=0;i<output.length;i++){
			Assert.assertEquals(output[i], d[i], 0.001f);
		}
	}
	
	@Test
	public void testSpatialConvolutionFull(){
		// given tensors created with Torch7
		float[] input = new float[]{0.19151945412159f, 0.49766367673874f, 0.62210875749588f, 0.81783843040466f, 0.43772774934769f, 0.61211186647415f, 0.78535860776901f, 0.77135992050171f, 0.77997583150864f, 0.86066979169846f, 0.27259260416031f, 0.15063697099686f, 0.27646425366402f, 0.19851875305176f, 0.80187219381332f, 0.81516295671463f, 0.95813935995102f, 0.15881535410881f, 0.8759326338768f, 0.11613783240318f, 0.35781726241112f, 0.012907532975078f, 0.50099509954453f, 0.48683345317841f, 0.68346291780472f,};
		T t = factory.createTensor(input, 1,5,5);
		float[] kernel = new float[]{-0.32415437698364f, -0.31857073307037f, 0.18188442289829f, -0.13951431214809f, 0.25509414076805f, -0.1690703779459f, -0.090076014399529f, 0.15885785222054f, 0.076930783689022f,};
		T k = factory.createTensor(kernel, 1, 1, 3, 3);
		T b = factory.createTensor(1); // 0 bias
		b.fill(0.0f);
		float[] output = new float[]{-0.10229878127575f, -0.20980164408684f, -0.36842292547226f, -0.26365938782692f, -0.33619004487991f, -0.33169940114021f, 0.026133447885513f, -0.22617430984974f, 0.25943052768707f,};
		T exp = factory.createTensor(output, 1, 3, 3);
		
		T res = math.spatialconvolve(null, b, t, k, 1, 1, 0, 0);
		Assert.assertEquals(true, res.sameDim(exp));
		float[] d = res.get();
		for(int i=0;i<output.length;i++){
			Assert.assertEquals(output[i], d[i], 0.001f);
		}
	}
	
	/**
	 * // TODO check ignoring for now 
	 */
	@Ignore
	@Test
	public void testSpatialConvolutionFullPadded1(){
		// given tensors created with Torch7
		float[] input = new float[]{0.19151945412159f, 0.49766367673874f, 0.62210875749588f, 0.81783843040466f, 0.43772774934769f, 0.61211186647415f, 0.78535860776901f, 0.77135992050171f, 0.77997583150864f, 0.86066979169846f, 0.27259260416031f, 0.15063697099686f, 0.27646425366402f, 0.19851875305176f, 0.80187219381332f, 0.81516295671463f, 0.95813935995102f, 0.15881535410881f, 0.8759326338768f, 0.11613783240318f, 0.35781726241112f, 0.012907532975078f, 0.50099509954453f, 0.48683345317841f, 0.68346291780472f,};
		T t = factory.createTensor(input, 1,5,5);
		float[] kernel = new float[]{-0.32415437698364f, -0.31857073307037f, 0.18188442289829f, -0.13951431214809f, 0.25509414076805f, -0.1690703779459f, -0.090076014399529f, 0.15885785222054f, 0.076930783689022f,};
		T k = factory.createTensor(kernel, 1, 1, 3, 3);
		T b = factory.createTensor(1); // 0 bias
		float[] output = new float[]{0.15499959886074f, 0.16358412802219f, 0.21280115842819f, 0.18517228960991f, -0.0097887143492699f, 0.2304742783308f, 0.25303274393082f, 0.27846127748489f, 0.34164065122604f, 0.21174050867558f, 0.033477135002613f, -0.040496986359358f, 0.016157247126102f, 0.1608159840107f, 0.10138539969921f, 0.31606841087341f, 0.21850445866585f, 0.22664746642113f, 0.12060117721558f, -0.035617150366306f, -0.019975544884801f, 0.14983840286732f, 0.22274658083916f, 0.21933452785015f, 0.22867679595947f,};
		T exp = factory.createTensor(output, 1, 5, 5);
		T padded = math.zeropad(null, t, 0, 1, 1);
		T res = math.spatialconvolve(null, b, padded, k, 1, 1, 0, 0);
		
		Assert.assertEquals(true, res.sameDim(exp));
		Assert.assertArrayEquals(output, res.get(), 0.001f);
	}
	
	/**
	 * // TODO check ignoring for now 
	 */
	@Ignore
	@Test
	public void testSpatialConvolutionFullPadded2(){
		// given tensors created with Torch7
		float[] input = new float[]{0.19151945412159f, 0.49766367673874f, 0.62210875749588f, 0.81783843040466f, 0.43772774934769f, 0.61211186647415f, 0.78535860776901f, 0.77135992050171f, 0.77997583150864f, 0.86066979169846f, 0.27259260416031f, 0.15063697099686f, 0.27646425366402f, 0.19851875305176f, 0.80187219381332f, 0.81516295671463f, 0.95813935995102f, 0.15881535410881f, 0.8759326338768f, 0.11613783240318f, 0.35781726241112f, 0.012907532975078f, 0.50099509954453f, 0.48683345317841f, 0.68346291780472f,};
		T t = factory.createTensor(input, 1,5,5);
		float[] kernel = new float[]{-0.32415437698364f, -0.31857073307037f, 0.18188442289829f, -0.13951431214809f, 0.25509414076805f, -0.1690703779459f, -0.090076014399529f, 0.15885785222054f, 0.076930783689022f,};
		T k = factory.createTensor(kernel, 1, 1, 3, 3);
		T b = factory.createTensor(1); // 0 bias
		float[] output = new float[]{0.15499959886074f, 0.16358412802219f, 0.21280115842819f, 0.18517228960991f, -0.0097887143492699f, 0.2304742783308f, 0.25303274393082f, 0.27846127748489f, 0.34164065122604f, 0.21174050867558f, 0.033477135002613f, -0.040496986359358f, 0.016157247126102f, 0.1608159840107f, 0.10138539969921f, 0.31606841087341f, 0.21850445866585f, 0.22664746642113f, 0.12060117721558f, -0.035617150366306f, -0.019975544884801f, 0.14983840286732f, 0.22274658083916f, 0.21933452785015f, 0.22867679595947f,};
		T exp = factory.createTensor(output, 1, 5, 5);
		
		T res = math.spatialconvolve(null, b, t, k, 1, 1, 1, 1);
		Assert.assertEquals(true, res.sameDim(exp));
		Assert.assertArrayEquals(output, res.get(), 0.001f);
	}
	
	@Test
	public void testSpatialConvolutionFullPadded3(){
		// given tensors created with Torch7
		float[] input = new float[]{0.19151945412159f, 0.49766367673874f, 0.62210875749588f, 0.81783843040466f, 0.43772774934769f, 0.61211186647415f, 0.78535860776901f, 0.77135992050171f, 0.77997583150864f, 0.86066979169846f, 0.27259260416031f, 0.15063697099686f, 0.27646425366402f, 0.19851875305176f, 0.80187219381332f, 0.81516295671463f, 0.95813935995102f, 0.15881535410881f, 0.8759326338768f, 0.11613783240318f, 0.35781726241112f, 0.012907532975078f, 0.50099509954453f, 0.48683345317841f, 0.68346291780472f,};
		T t = factory.createTensor(input, 1,5,5);
		float[] kernel = new float[]{-0.32415437698364f, -0.31857073307037f, 0.18188442289829f, -0.13951431214809f, 0.25509414076805f, -0.1690703779459f, -0.090076014399529f, 0.15885785222054f, 0.076930783689022f,};
		T k = factory.createTensor(kernel, 1, 1, 3, 3);
		T b = factory.createTensor(1); // 0 bias
		float[] output = new float[]{0.15499959886074f, 0.16358412802219f, 0.21280115842819f, 0.18517228960991f, -0.0097887143492699f, 0.2304742783308f, 0.25303274393082f, 0.27846127748489f, 0.34164065122604f, 0.21174050867558f, 0.033477135002613f, -0.040496986359358f, 0.016157247126102f, 0.1608159840107f, 0.10138539969921f, 0.31606841087341f, 0.21850445866585f, 0.22664746642113f, 0.12060117721558f, -0.035617150366306f, -0.019975544884801f, 0.14983840286732f, 0.22274658083916f, 0.21933452785015f, 0.22867679595947f,};
		T exp = factory.createTensor(output, 1, 5, 5);
		// add zero padding for same convolution
		
		T padded = math.zeropad(null, t, 0, 1, 1);
		T res1 = math.spatialconvolve(null, b, padded, k, 1, 1, 0, 0);
		T res2 = math.spatialconvolve(null, b, t, k, 1, 1, 1, 1);
		Assert.assertEquals(res1, res2);
	}
	
	@Test
	public void testSpatialConvolutionAdd(){
		// given tensors created with Torch7
		float[] input = new float[]{0.19151945412159f, 0.49766367673874f, 0.62210875749588f, 0.81783843040466f, 0.43772774934769f, 0.61211186647415f, 0.78535860776901f, 0.77135992050171f, 0.77997583150864f, 0.86066979169846f, 0.27259260416031f, 0.15063697099686f, 0.27646425366402f, 0.19851875305176f, 0.80187219381332f, 0.81516295671463f, 0.95813935995102f, 0.15881535410881f, 0.8759326338768f, 0.11613783240318f, 0.35781726241112f, 0.012907532975078f, 0.50099509954453f, 0.48683345317841f, 0.68346291780472f,};
		T t = factory.createTensor(input, 1,5,5);
		float[] kernel = new float[]{-0.32415437698364f, -0.31857073307037f, 0.18188442289829f, -0.13951431214809f, 0.25509414076805f, -0.1690703779459f, -0.090076014399529f, 0.15885785222054f, 0.076930783689022f,};
		T k = factory.createTensor(kernel, 1, 1, 3, 3);
		T b = factory.createTensor(1); // 0 bias
		b.fill(2.0f);
		float[] output = new float[]{1.8977012634277f, 1.7901983261108f, 1.6315770149231f, 1.7363406419754f, 1.6638100147247f, 1.6683006286621f, 2.0261335372925f, 1.7738256454468f, 2.2594304084778f, };
		T exp = factory.createTensor(output, 1, 3, 3);
		
		T res = math.spatialconvolve(null, b, t, k, 1, 1, 0, 0);
		Assert.assertEquals(true, res.sameDim(exp));
		float[] d = res.get();
		for(int i=0;i<output.length;i++){
			Assert.assertEquals(output[i], d[i], 0.001f);
		}
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
	public void testSpatialDinConvolution(){
		// given tensors created with Torch7
		float[] gradOutput = new float[]{0.91212290525436f, 0.47385999560356f, 0.79052412509918f, 0.068084716796875f, 0.99208146333694f, 0.38271069526672f, 0.95880174636841f, 0.11855413764715f, 0.79196411371231f, 0.89632850885391f, 0.28525096178055f, 0.76430851221085f, 0.62491673231125f, 0.37540492415428f, 0.47809380292892f, 0.025814190506935f, 0.1956751793623f, 0.90653091669083f, 0.38231745362282f, 0.78641778230667f, 0.053873687982559f, 0.61931520700455f, 0.45164841413498f, 0.21158893406391f, 0.98200476169586f, 0.29230400919914f, 0.12394270300865f, 0.6615446805954f, 0.11938089877367f, 0.88626098632812f, 0.73852306604385f, 0.67117643356323f, };
		T g = factory.createTensor(gradOutput, 2,4,4);
		float[] kernel = new float[]{-0.13256266713142f, -0.081123024225235f, 0.20028452575207f, 0.14779895544052f, -0.027275109663606f, 0.06908542662859f, 0.19295339286327f, -0.21334519982338f, -0.20750792324543f, 0.23332524299622f, -0.14882849156857f, 0.089206539094448f, -0.21337877213955f, 0.20248989760876f, 0.082439668476582f, 0.19710204005241f, 0.044606547802687f, 0.2240593880415f, 0.015702562406659f, -0.048553735017776f, -0.21527910232544f, -0.11189913004637f, 0.028959833085537f, -0.032926961779594f, -0.080295063555241f, 0.12470100075006f, 0.0013985770056024f, 0.047013558447361f, -0.18295477330685f, -0.19754350185394f, 0.050531595945358f, 0.09642318636179f, 0.031086603179574f, -0.1583856344223f, -0.23251365125179f, -0.22045263648033f, };
		T k = factory.createTensor(kernel, 2, 2, 3, 3);
		float[] gradInput = new float[]{-0.11784084141254f, -0.13207621872425f, -0.040687724947929f, -0.17961969971657f, 0.032318040728569f, -0.15566299855709f, -0.017752096056938f, -0.174720287323f, 0.14993214607239f, -0.25431695580482f, 0.13785444200039f, -0.042996801435947f, 0.21132075786591f, -0.41594505310059f, -0.065341003239155f, -0.28472328186035f, -0.089376360177994f, -0.0011424487456679f, 0.11329964548349f, -0.16655434668064f, -0.052588686347008f, -0.31698375940323f, -0.28121370077133f, -0.13260515034199f, 0.15296545624733f, 0.04569948464632f, -0.22996382415295f, -0.20281861722469f, -0.11213970184326f, -0.17799127101898f, 0.11099410057068f, -0.11716279387474f, -0.066131353378296f, -0.13547566533089f, -0.01998633146286f, -0.0044179568067193f, 0.22202068567276f, -0.01836684346199f, 0.0087583027780056f, -0.27154901623726f, -0.15901625156403f, -0.14927811920643f, 0.049270555377007f, 0.10916593670845f, 0.26781758666039f, 0.013473942875862f, 0.10662660747766f, -0.0011622738093138f, 0.1707751005888f, 0.026069637387991f, -0.13207778334618f, 0.18197172880173f, -0.18330423533916f, -0.20426425337791f, 0.2190637588501f, 0.10204625874758f, 0.41224229335785f, -0.41041719913483f, 0.08739622682333f, -0.066791400313377f, -0.12674915790558f, 0.040099829435349f, 0.12157627940178f, 0.42109137773514f, 0.049180809408426f, 0.048403926193714f, 0.10426414757967f, -0.066260240972042f, -0.098359793424606f, -0.36287280917168f, -0.21059414744377f, -0.1421786993742f, };
		T exp = factory.createTensor(gradInput, 2, 6, 6);
				
		T res = math.spatialdinconvolve(null, g, k, 1, 1, 0, 0);
		Assert.assertEquals(true, res.sameDim(exp));
		float[] d = res.get();
		for(int i=0;i<gradInput.length;i++){
			Assert.assertEquals(gradInput[i], d[i], 0.001f);
		}
	}
	
	@Test
	public void testSpatialDkerConvolution(){
		// given tensors created with Torch7
		
		float[] gradOutput = new float[]{0.91212290525436f, 0.47385999560356f, 0.79052412509918f, 0.068084716796875f, 0.99208146333694f, 0.38271069526672f, 0.95880174636841f, 0.11855413764715f, 0.79196411371231f, 0.89632850885391f, 0.28525096178055f, 0.76430851221085f, 0.62491673231125f, 0.37540492415428f, 0.47809380292892f, 0.025814190506935f, 0.1956751793623f, 0.90653091669083f, 0.38231745362282f, 0.78641778230667f, 0.053873687982559f, 0.61931520700455f, 0.45164841413498f, 0.21158893406391f, 0.98200476169586f, 0.29230400919914f, 0.12394270300865f, 0.6615446805954f, 0.11938089877367f, 0.88626098632812f, 0.73852306604385f, 0.67117643356323f, };
		T g = factory.createTensor(gradOutput, 2,4,4);
		float[] input = new float[]{0.19151945412159f, 0.49766367673874f, 0.62210875749588f, 0.81783843040466f, 0.43772774934769f, 0.61211186647415f, 0.78535860776901f, 0.77135992050171f, 0.77997583150864f, 0.86066979169846f, 0.27259260416031f, 0.15063697099686f, 0.27646425366402f, 0.19851875305176f, 0.80187219381332f, 0.81516295671463f, 0.95813935995102f, 0.15881535410881f, 0.8759326338768f, 0.11613783240318f, 0.35781726241112f, 0.012907532975078f, 0.50099509954453f, 0.48683345317841f, 0.68346291780472f, 0.33101543784142f, 0.71270203590393f, 0.80263954401016f, 0.37025076150894f, 0.098251938819885f, 0.56119620800018f, 0.0559934489429f, 0.5030831694603f, 0.44266265630722f, 0.013768452219665f, 0.022143909707665f, 0.7728266119957f, 0.29072853922844f, 0.88264119625092f, 0.24639444053173f, 0.36488598585129f, 0.73828679323196f, 0.61539614200592f, 0.88922613859177f, 0.075381234288216f, 0.98713928461075f, 0.36882400512695f, 0.11744338274002f, 0.93314009904861f, 0.39378234744072f, 0.65137815475464f, 0.45272982120514f, 0.39720258116722f, 0.53814780712128f, 0.78873014450073f, 0.79062211513519f, 0.31683611869812f, 0.46583634614944f, 0.56809866428375f, 0.43533223867416f, 0.86912739276886f, 0.56947863101959f, 0.43617340922356f, 0.96925902366638f, 0.80214762687683f, 0.040556151419878f, 0.14376682043076f, 0.54811954498291f, 0.70426094532013f, 0.462576597929f, 0.7045813202858f, 0.37647223472595f, };
		T t = factory.createTensor(input, 2,6,6);
		
		// TODO is this correct???
		T b = factory.createTensor(2,2,3,3); // 0 bias
		b.fill(0);
		
		float[] gradWeights = new float[]{4.8932576179504f, 5.4131393432617f, 4.9660415649414f, 4.7953443527222f, 4.8381161689758f, 5.1261992454529f, 4.5720977783203f, 3.4433665275574f, 4.7362084388733f, 5.2743463516235f, 4.9831132888794f, 4.1826686859131f, 5.634349822998f, 5.2543325424194f, 4.284435749054f, 5.7358064651489f, 5.0501775741577f, 4.6825685501099f, 3.8726243972778f, 4.1302518844604f, 4.2423276901245f, 4.8874883651733f, 4.5437321662903f, 4.0450367927551f, 3.6376442909241f, 3.8371171951294f, 3.2664217948914f, 4.5259666442871f, 3.8531394004822f, 4.2626647949219f, 5.3499975204468f, 4.6342439651489f, 4.0881423950195f, 4.9671773910522f, 4.5640454292297f, 3.8721601963043f, };
		T exp = factory.createTensor(gradWeights, 2,2,3,3);
		
		T res = math.spatialdkerconvolve(null, b, g, t, 1, 1, 0, 0);
		Assert.assertEquals(true, res.sameDim(exp));
		float[] d = res.get();
		for(int i=0;i<gradWeights.length;i++){
			Assert.assertEquals(gradWeights[i], d[i], 0.001f);
		}
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
	public void testSpatialDMaxpool(){
		float[] data = new float[]{1.0f, 0.0f,
									0.2f, 1.0f,
									
									0.5f, 1.0f,
									0.1f, 0.0f};
		float[] expData = new float[]{2.0f, 0.0f,
									0.0f, 0.0f,
									
									0.0f, 2.0f,
									0.0f, 0.0f};
		T t1 = factory.createTensor(data, 2, 2, 2);
		T t2 = factory.createTensor(2, 1, 1);
		t2.fill(2.0f);
		T res = math.spatialdmaxpool(null, t2, t1, 2, 2, 2, 2);
		Assert.assertArrayEquals(expData, res.get(), 0.001f);
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
