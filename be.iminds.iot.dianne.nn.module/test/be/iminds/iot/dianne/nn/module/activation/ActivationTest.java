package be.iminds.iot.dianne.nn.module.activation;

import java.util.Arrays;
import java.util.Collection;
import java.util.UUID;

import junit.framework.Assert;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import be.iminds.iot.dianne.api.nn.module.BackwardListener;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.ModuleException;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.dianne.tensor.impl.java.JavaTensorFactory;
import be.iminds.iot.dianne.tensor.impl.th.THTensorFactory;

// These tests are based on input/output of corresponding Torch7 modules
@RunWith(Parameterized.class)
public class ActivationTest {

	private TensorFactory factory;

	public ActivationTest(TensorFactory f, String name) {
		this.factory = f;
	}

	@Parameters(name="{1}")
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] { 
				{ new JavaTensorFactory(), "Java Tensor" },
				{ new THTensorFactory(), "TH Tensor" }
		});
	}

	private void testActivation(Module m, Tensor input, Tensor gradOutput,
			Tensor expOutput, Tensor expGradInput) throws Exception {
		final Tensor out = factory.createTensor(expOutput.dims());
		final Tensor gradIn = factory.createTensor(expGradInput.dims());

		m.addForwardListener(new ForwardListener() {
			@Override
			public void onForward(UUID moduleId, Tensor output, String... tags) {
				System.out.println(output);
				output.copyInto(out);

				m.backward(UUID.randomUUID(), gradOutput);
			}

			@Override
			public void onError(UUID moduleId, ModuleException e, String... tags) {
				e.printStackTrace();
			}
		});

		m.addBackwardListener(new BackwardListener() {
			@Override
			public void onBackward(UUID moduleId, Tensor gradInput, String... tags) {
				System.out.println(gradInput);
				gradInput.copyInto(gradIn);
				synchronized (m) {
					m.notify();
				}
			}

			@Override
			public void onError(UUID moduleId, ModuleException e, String... tags) {
				e.printStackTrace();
			}
		});

		m.forward(UUID.randomUUID(), input);
		synchronized (m) {
			m.wait(1000);
		}

		for (int i = 0; i < expOutput.size(); i++) {
			Assert.assertEquals(expOutput.get()[i], out.get()[i], 0.001f);
		}
		for (int i = 0; i < expGradInput.size(); i++) {
			Assert.assertEquals(expGradInput.get()[i], gradIn.get()[i], 0.001f);
		}
	}

	private Tensor input;
	private Tensor gradOutput;

	@Before
	public void setup() {
		input = factory.createTensor(11);
		int v = -5;
		for (int i = 0; i < 11; i++) {
			input.set(v, i);
			v++;
		}

		gradOutput = factory.createTensor(11);
		gradOutput.fill(1.0f);
	}

	@Test
	public void testSigmoid() throws Exception {
		float[] eo = new float[] { 0.0067f, 0.0180f, 0.0474f, 0.1192f, 0.2689f,
				0.5000f, 0.7311f, 0.8808f, 0.9526f, 0.9820f, 0.9933f };
		Tensor expOutput = factory.createTensor(eo, 11);

		float[] eg = new float[] { 0.0066f, 0.0177f, 0.0452f, 0.1050f, 0.1966f,
				0.2500f, 0.1966f, 0.1050f, 0.0452f, 0.0177f, 0.0066f };
		Tensor expGradInput = factory.createTensor(eg, 11);

		Module m = new Sigmoid(factory);

		testActivation(m, input, gradOutput, expOutput, expGradInput);
	}

	@Test
	public void testTanh() throws Exception {
		float[] eo = new float[] { -0.9999f, -0.9993f, -0.9951f, -0.9640f,
				-0.7616f, 0.0000f, 0.7616f, 0.9640f, 0.9951f, 0.9993f, 0.9999f };
		Tensor expOutput = factory.createTensor(eo, 11);

		float[] eg = new float[] { 0.0002f, 0.0013f, 0.0099f, 0.0707f, 0.4200f,
				1.0000f, 0.4200f, 0.0707f, 0.0099f, 0.0013f, 0.0002f };

		Tensor expGradInput = factory.createTensor(eg, 11);

		Module m = new Tanh(factory);

		testActivation(m, input, gradOutput, expOutput, expGradInput);
	}

	@Test
	public void testReLU() throws Exception {
		float[] eo = new float[] { 0f, 0f, 0f, 0f, 0f, 0f, 1f, 2f, 3f, 4f, 5f };
		Tensor expOutput = factory.createTensor(eo, 11);

		float[] eg = new float[] { 0f, 0f, 0f, 0f, 0f, 0f, 1f, 1f, 1f, 1f, 1f };

		Tensor expGradInput = factory.createTensor(eg, 11);

		Module m = new ReLU(factory);

		testActivation(m, input, gradOutput, expOutput, expGradInput);
	}
	
	@Test
	public void testPReLU() throws Exception {
		float[] eo = new float[] { -0.5f, -0.4f, -0.3f, -0.2f, -0.1f, 0f, 1f, 2f, 3f, 4f, 5f };
		Tensor expOutput = factory.createTensor(eo, 11);

		float[] eg = new float[] { 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 1f, 1f, 1f, 1f, 1f };

		Tensor expGradInput = factory.createTensor(eg, 11);

		Module m = new PReLU(factory, 0.1f);

		testActivation(m, input, gradOutput, expOutput, expGradInput);
	}

	@Test
	public void testSoftmax() throws Exception {
		float[] eo = new float[] { 0.0000f, 0.0001f, 0.0002f, 0.0006f, 0.0016f,
				0.0043f, 0.0116f, 0.0315f, 0.0855f, 0.2325f, 0.6321f };
		Tensor expOutput = factory.createTensor(eo, 11);

		float[] eg = new float[] { 0.0003e-17f, 0.0009e-17f, 0.0024e-17f,
				0.0065e-17f, 0.0175e-17f, 0.0475e-17f, 0.1287e-17f,
				0.3495e-17f, 0.9498e-17f, 2.5816e-17f, 7.0175e-17f };

		Tensor expGradInput = factory.createTensor(eg, 11);

		Module m = new Softmax(factory);

		testActivation(m, input, gradOutput, expOutput, expGradInput);
	}
}
