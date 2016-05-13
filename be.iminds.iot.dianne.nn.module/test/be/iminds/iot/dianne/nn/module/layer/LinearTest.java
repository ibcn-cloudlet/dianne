package be.iminds.iot.dianne.nn.module.layer;

import org.junit.Test;

import be.iminds.iot.dianne.nn.module.ModuleTest;
import be.iminds.iot.dianne.tensor.Tensor;

public class LinearTest extends ModuleTest {

	@Test
	public void testLinear() throws Exception {
		Linear l = new Linear(6, 3);
		l.getParameters().rand();

		Tensor params = new Tensor(new float[]{0.4653371f, 0.8163113f, 0.09029245f, 0.16172619f, 0.1672737f, 
		                                        0.54460245f, 0.23656675f, 0.82285637f, 0.29915535f, 0.88890827f, 
		                                        0.45202088f, 0.28761277f, 0.47259474f, 0.920486f, 0.389602f,
		                                        0.6109357f, 0.53176844f, 0.69206583f, 0.83310753f, 0.709608f, 0.10931859f}, 21);
		Tensor input = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, 6);
		Tensor gradOutput = new Tensor(new float[]{0.1f, 0.2f, 0.3f}, 3);
		
		Tensor expOutput = new Tensor(new float[]{1.5450801f, 1.741724f, 1.3830539f},3);
		Tensor expGradInput = new Tensor(new float[]{0.23562549f, 0.5223482f, 0.18574092f, 0.377235f, 0.2666621f, 0.31960255f}, 6);
		Tensor expDeltaParams = new Tensor(new float[]{0.01f, 0.02f, 0.03f, 0.04f, 0.05f,
														0.06f, 0.02f, 0.04f, 0.06f, 0.08f, 
														0.1f, 0.12f, 0.03f, 0.06f, 0.09f, 
														0.12f, 0.15f, 0.18f, 0.1f, 0.2f, 0.3f}, 21);
		
		testModule(l, params, input, expOutput, gradOutput, expGradInput, expDeltaParams);
	}
	
	@Test
	public void testLinearBatch() throws Exception {
		Linear l = new Linear(6, 3);
		l.getParameters().rand();

		Tensor params = new Tensor(new float[]{0.4653371f, 0.8163113f, 0.09029245f, 0.16172619f, 0.1672737f, 
		                                        0.54460245f, 0.23656675f, 0.82285637f, 0.29915535f, 0.88890827f, 
		                                        0.45202088f, 0.28761277f, 0.47259474f, 0.920486f, 0.389602f,
		                                        0.6109357f, 0.53176844f, 0.69206583f, 0.83310753f, 0.709608f, 0.10931859f}, 21);
		Tensor input = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
											  0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, 2, 6);
		Tensor gradOutput = new Tensor(new float[]{0.1f, 0.2f, 0.3f,
												  0.1f, 0.2f, 0.3f}, 2, 3);
		
		Tensor expOutput = new Tensor(new float[]{1.5450801f, 1.741724f, 1.3830539f,
												  1.5450801f, 1.741724f, 1.3830539f}, 2, 3);
		Tensor expGradInput = new Tensor(new float[]{0.23562549f, 0.5223482f, 0.18574092f, 0.377235f, 0.2666621f, 0.31960255f,
													 0.23562549f, 0.5223482f, 0.18574092f, 0.377235f, 0.2666621f, 0.31960255f}, 2, 6);
		Tensor expDeltaParams = new Tensor(new float[]{0.02f, 0.04f, 0.06f, 0.08f, 0.1f,
														0.12f, 0.04f, 0.08f, 0.12f, 0.16f, 
														0.2f, 0.24f, 0.06f, 0.12f, 0.18f, 
														0.24f, 0.3f, 0.36f, 0.2f, 0.4f, 0.6f}, 21);
		
		testModule(l, params, input, expOutput, gradOutput, expGradInput, expDeltaParams);
	}
	
	
	/**
	 * Test a linear with 3d input as from convolutional layer
	 * @throws Exception
	 */
	@Test
	public void testLinearFrom3D() throws Exception {
		Linear l = new Linear(6, 3);
		l.getParameters().rand();

		Tensor params = new Tensor(new float[]{0.4653371f, 0.8163113f, 0.09029245f, 0.16172619f, 0.1672737f, 
		                                        0.54460245f, 0.23656675f, 0.82285637f, 0.29915535f, 0.88890827f, 
		                                        0.45202088f, 0.28761277f, 0.47259474f, 0.920486f, 0.389602f,
		                                        0.6109357f, 0.53176844f, 0.69206583f, 0.83310753f, 0.709608f, 0.10931859f}, 21);
		Tensor input = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, 1, 2, 3);
		Tensor gradOutput = new Tensor(new float[]{0.1f, 0.2f, 0.3f}, 3);
		
		Tensor expOutput = new Tensor(new float[]{1.5450801f, 1.741724f, 1.3830539f},3);
		Tensor expGradInput = new Tensor(new float[]{0.23562549f, 0.5223482f, 0.18574092f, 0.377235f, 0.2666621f, 0.31960255f}, 1, 2, 3);
		Tensor expDeltaParams = new Tensor(new float[]{0.01f, 0.02f, 0.03f, 0.04f, 0.05f,
														0.06f, 0.02f, 0.04f, 0.06f, 0.08f, 
														0.1f, 0.12f, 0.03f, 0.06f, 0.09f, 
														0.12f, 0.15f, 0.18f, 0.1f, 0.2f, 0.3f}, 21);
		
		testModule(l, params, input, expOutput, gradOutput, expGradInput, expDeltaParams);
	}
	
	/**
	 * Test a linear with 3d input as from batched convolutional layer
	 * @throws Exception
	 */
	@Test
	public void testLinearFrom4D() throws Exception {
		Linear l = new Linear(6, 3);
		l.getParameters().rand();

		Tensor params = new Tensor(new float[]{0.4653371f, 0.8163113f, 0.09029245f, 0.16172619f, 0.1672737f, 
		                                        0.54460245f, 0.23656675f, 0.82285637f, 0.29915535f, 0.88890827f, 
		                                        0.45202088f, 0.28761277f, 0.47259474f, 0.920486f, 0.389602f,
		                                        0.6109357f, 0.53176844f, 0.69206583f, 0.83310753f, 0.709608f, 0.10931859f}, 21);
		Tensor input = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
											  0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, 2, 1, 2, 3);
		Tensor gradOutput = new Tensor(new float[]{0.1f, 0.2f, 0.3f,
												  0.1f, 0.2f, 0.3f}, 2, 3);
		
		Tensor expOutput = new Tensor(new float[]{1.5450801f, 1.741724f, 1.3830539f,
												  1.5450801f, 1.741724f, 1.3830539f}, 2, 3);
		Tensor expGradInput = new Tensor(new float[]{0.23562549f, 0.5223482f, 0.18574092f, 0.377235f, 0.2666621f, 0.31960255f,
													 0.23562549f, 0.5223482f, 0.18574092f, 0.377235f, 0.2666621f, 0.31960255f}, 2, 1, 2, 3);
		Tensor expDeltaParams = new Tensor(new float[]{0.02f, 0.04f, 0.06f, 0.08f, 0.1f,
														0.12f, 0.04f, 0.08f, 0.12f, 0.16f, 
														0.2f, 0.24f, 0.06f, 0.12f, 0.18f, 
														0.24f, 0.3f, 0.36f, 0.2f, 0.4f, 0.6f}, 21);
		
		testModule(l, params, input, expOutput, gradOutput, expGradInput, expDeltaParams);
	}
}
