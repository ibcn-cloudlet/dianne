package be.iminds.iot.dianne.nn.module.layer;

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
		Tensor expDeltaParams = new Tensor(new float[]{0.007f, 0.009000001f, 0.011000001f, 0.013f, 0.015000001f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 20);

		testModule(bn, params, input, expOutput, gradOutput, expGradInput, expDeltaParams);
	}
	
}
