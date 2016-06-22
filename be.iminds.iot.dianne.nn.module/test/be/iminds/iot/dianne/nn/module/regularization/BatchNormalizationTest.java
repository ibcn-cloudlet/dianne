package be.iminds.iot.dianne.nn.module.regularization;

import org.junit.Assert;
import org.junit.Test;

import be.iminds.iot.dianne.nn.module.ModuleTest;
import be.iminds.iot.dianne.tensor.Tensor;

public class BatchNormalizationTest extends ModuleTest {

	@Test
	public void testBatchNormalization() throws Exception {
		
		BatchNormalization bn = new BatchNormalization(5);
		
		Tensor params = new Tensor(20);
		params.fill(0.0f);
		params.narrow(0, 0, 5).fill(0.5f);
		params.narrow(0, 5, 5).fill(0.1f);
		params.narrow(0, 15, 5).fill(1.0f);
		
		Tensor input = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f},2, 5);
		
		Tensor gradOutput = new Tensor(2, 5);
		gradOutput.fill(0.1f);

		// TODO check if these are the correct values?
		Tensor expOutput = new Tensor(new float[]{0.14999975f, 0.19999951f, 0.24999925f, 0.299999f, 0.34999874f, 0.39999852f, 0.44999823f, 0.499998f, 0.54999775f, 0.5999975f},2, 5);
		Tensor expGradInput = new Tensor(new float[]{-0.00175f, -0.0045000003f, -0.008250001f, -0.012999999f, -0.01875f, -0.010500001f, -0.01575f, -0.022000002f, -0.029249998f, -0.0375f}, 2, 5);
		Tensor expDeltaParams = new Tensor(new float[]{0.07f, 0.09000001f, 0.11000001f, 0.13f, 0.15000001f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 20);

		testModule(bn, params, input, expOutput, gradOutput, expGradInput, expDeltaParams);
	}

	
	@Test
	public void testSpatialBatchNormalization3d() throws Exception {
		BatchNormalization bn = new BatchNormalization(2);
		
		Tensor params = new Tensor(8);
		params.fill(0.0f);
		params.narrow(0, 0, 2).fill(0.5f);
		params.narrow(0, 2, 2).fill(0.1f);
		params.narrow(0, 6, 2).fill(1.0f);
		
		Tensor input = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}, 2, 2, 2);
		
		// just test forward of 1 sample
		Tensor gradOutput = new Tensor(2, 2, 2);
		gradOutput.fill(0.1f);

		// TODO check if these are the correct values?
		Tensor expOutput = new Tensor(new float[]{0.14999975f, 0.19999951f, 0.24999925f, 0.299999f, 0.34999874f, 0.39999852f, 0.44999823f, 0.499998f}, 2 ,2 ,2);

		testModule(bn, params, input, expOutput, gradOutput, null, null);
	}
	
	@Test
	public void testSpatialBatchNormalization4d() throws Exception {
		BatchNormalization bn = new BatchNormalization(2);
		bn.train = true;
		
		Tensor params = new Tensor(new float[]{0.5f, 0.5f, 0f, 0f, 0f, 0f, 1f, 1f}, 8);
		
		Tensor input = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f},2, 2, 2, 2);
		
		Tensor gradOutput = new Tensor(2, 2, 2, 2);
		gradOutput.fill(0.1f);

		// TODO check if these are the correct values?
		Tensor expOutput = new Tensor(new float[]{-0.66210306f, -0.5417207f, -0.42133832f, -0.30095595f, -0.662103f, -0.54172057f, -0.42133826f, -0.30095586f, 0.30095586f, 0.42133826f, 0.5417206f, 0.66210306f, 0.30095592f, 0.42133832f, 0.5417207f, 0.6621031f},2 , 2 ,2 ,2);
		// What should we get as gradinput?!
		Tensor expGradInput = new Tensor(2, 2, 2, 2);
		expGradInput.fill(0.0f);
		Tensor expDeltaParams = new Tensor(new float[]{0f, 0f, 0.8f, 0.8f, 0.0f, 0.0f, 0.0f, 0.0f}, 8);

		testModule(bn, params, input, expOutput, gradOutput, expGradInput, expDeltaParams);
		
		float[] p = bn.getParameters().get();
		// check updated mean and var
		Assert.assertEquals(p[4], 0.065f, 0.0001f);
		Assert.assertEquals(p[5], 0.105f, 0.0001f);
		Assert.assertEquals(p[6], 0.9197f, 0.0001f);
		Assert.assertEquals(p[7], 0.9197f, 0.0001f);
	}
}
