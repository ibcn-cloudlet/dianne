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
		
		Tensor input = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f},2, 5);
		
		Tensor gradOutput = new Tensor(2, 5);
		gradOutput.fill(0.1f);

		// TODO check if these are the correct values - seems ok...
		Tensor expOutput = new Tensor(new float[]{-0.39996004f, -0.39995998f, -0.39996f, -0.39995998f, -0.39996f, 0.59995997f, 0.59996f, 0.59996f, 0.59996f, 0.59996f},2, 5);
		Tensor expGradInput = new Tensor(new float[]{-1.48975845E-8f, 7.4487922E-9f, 0.0f, 1.4897586E-8f, 0.0f, 1.4897583E-8f, -7.4487927E-9f, 0.0f, -1.4897588E-8f, 0.0f}, 2, 5);
		Tensor expDeltaParams = new Tensor(new float[]{-1.4899968E-9f, 7.4499845E-10f, 0.0f, 1.489997E-9f, 0.0f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 20);

		testModule(bn, params, input, expOutput, gradOutput, expGradInput, expDeltaParams);
	}
	
}
