package be.iminds.iot.dianne.nn.module.layer;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;
import java.util.Collection;
import java.util.UUID;

import javax.imageio.ImageIO;

import junit.framework.Assert;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import be.iminds.iot.dianne.api.nn.module.BackwardListener;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.ModuleException;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.dianne.tensor.impl.java.JavaTensorFactory;
import be.iminds.iot.dianne.tensor.impl.th.THTensorFactory;
import be.iminds.iot.dianne.tensor.util.ImageConverter;

@RunWith(Parameterized.class)
public class SpatialConvolutionTest {

	private TensorFactory factory;
	private ImageConverter converter;

	public SpatialConvolutionTest(TensorFactory f, String name) {
		this.factory = f;
		this.converter = new ImageConverter(f);
	}

	@Parameters(name="{1}")
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] { 
				{ new JavaTensorFactory(), "Java Tensor" },
				{ new THTensorFactory(), "TH Tensor" }
		});
	}
	
	@Test
	public void testSpatialConvolutionConstructor1() {
		int noInputPlanes = 3;
		int noOutputPlanes = 2;
		int kernelWidth = 3;
		int kernelHeight = 3;
		
		SpatialConvolution conv = new SpatialConvolution(factory, noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, 1 ,1, false);
	
		for(int i=0;i<noOutputPlanes;i++){
			Tensor sub1 = conv.weights.select(0, i);
			for(int j=0;j<noInputPlanes;j++){
				Tensor sub2 = sub1.select(0, j);
				System.out.println("Kernel:");
				sub2.fill(j);
				System.out.println(sub2);
				System.out.println("===");
			}
		}
		
		System.out.println(conv.getParameters());
	}
	
	// test based on randomly generated data with JavaTensor
	// TODO should be checked with for example Torch
	@Test
	public void testSpatialConvolution() throws InterruptedException {
		int noInputPlanes = 2;
		int noOutputPlanes = 2;
		int kernelWidth = 3;
		int kernelHeight = 3;
		
		SpatialConvolution conv = new SpatialConvolution(factory, noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, 1 ,1, false);
		conv.zeroDeltaParameters();
		
		float[] inputData = new float[]{-1.0016336f, -0.3659996f, -0.561763f, 0.025022956f, 0.39551026f, 
				0.25571227f, -0.36720142f, -1.1175866f, 1.6601679f, 0.7697076f, 
				0.94416225f, -0.33390528f, -0.7185745f, -0.82316816f, -0.4198505f, 
				0.8720639f, -0.91903704f, 1.1918385f, 0.9839494f, -1.1864451f, 
				-0.2240562f, 0.1604332f, -1.2869335f, -0.3002753f, 0.70621985f, 

				1.6712323f, 0.427771f, -1.1304947f, 1.5576284f, 0.8286627f, 
				1.1835989f, -1.261588f, 0.109402984f, 0.31725317f, 1.2378355f, 
				0.40925455f, -0.56402963f, -1.0522915f, -0.9422165f, 1.0155184f, 
				-0.31722248f, 1.1515416f, -0.14420497f, -1.6517973f, 0.4656332f, 
				1.050356f, -0.5541778f, -0.21497928f, -0.2700798f, 0.7216463f};
		
		Tensor input = factory.createTensor(inputData, 2,5,5);
		
		float[] gradOutputData = new float[]{
				0.014658666f, 0.019304574f, 0.037086185f, 
				0.04770545f, 0.0038461208f, 0.008546961f, 
				0.05205586f, 0.07534614f, 0.08972343f, 

				0.06575316f, 0.07568269f, 0.006832737f, 
				0.038957085f, 0.018131584f, 0.01639061f, 
				0.0061158002f, 0.069639005f, 0.059886318f};
		
		Tensor gradOutput = factory.createTensor(gradOutputData, 2,3,3);
		
		float[] paramsData = new float[]{0.06909947f, -0.11323614f, 0.21277206f, -0.12377161f, 0.1685688f, -0.08619809f, -0.019689009f, 0.023438826f, -0.07733278f, -0.19202958f, -0.13959476f, -0.22857352f, -0.07216856f, -0.020798072f, 0.055302367f, -0.18716966f, 0.21065162f, -0.04933682f, -0.22820695f, -0.12155786f, -0.044059023f, -0.009297177f, -0.20280977f, 0.15682052f, -0.21233399f, -0.22110622f, -0.122332945f, -0.22661619f, 0.12204148f, -0.093332455f, -0.11187785f, 0.13372453f, -0.097488046f, 0.18008412f, -0.05052042f, -0.018311232f, 0.097624585f, 0.11113395f};
		Tensor params = factory.createTensor(paramsData, paramsData.length);
		conv.setParameters(params);
		
		final Tensor output = factory.createTensor(2,3,3);
		final Tensor gradInput = factory.createTensor(2,5,5);
		
		conv.addForwardListener(new ForwardListener() {
			@Override
			public void onForward(UUID moduleId, Tensor o, String... tags) {
				o.copyInto(output);
//				System.out.println("OUTPUT CONV "+output);
				conv.backward(UUID.randomUUID(), gradOutput);
			}
		});
		
		conv.addBackwardListener(new BackwardListener() {
			@Override
			public void onBackward(UUID moduleId, Tensor gi, String... tags) {
				gi.copyInto(gradInput);
//				System.out.println("BACKWARD CONV "+gradInput);
			}
		});
		conv.forward(UUID.randomUUID(), input);
		
		Thread.sleep(200);
		conv.accGradParameters();

		
		float[] expOutput = new float[]{-0.33659852f, -0.335046f, 0.33185792f, 
				-0.16611992f, 0.5136749f, -0.46657073f, 
				-0.2812001f, 0.57139516f, 0.25588644f, 
				-0.14429103f, 0.7961464f, 0.5171676f, 
				-0.5985439f, 0.6174212f, -0.2471626f, 
				0.8638994f, 0.53555524f, -0.009491518f};
		
		float[] expGradInput = new float[]{-0.013992422f, -0.025590084f, -0.0101605095f, -0.004257101f, 0.007589861f, 
				-0.008019502f, -0.02796689f, -0.0050563263f, 0.012129665f, -0.001028845f, 
				-0.018315654f, -0.04847254f, -0.043538984f, -0.015240677f, 0.014581912f, 
				-0.015711082f, -0.01385968f, -0.032634832f, 0.0014659166f, -0.0010086568f, 
				-0.0023235206f, -0.016402332f, -0.032887846f, -0.025484096f, -0.014264632f, 

				-0.01771563f, -0.014879633f, -0.011615928f, -0.015819326f, -0.009114635f, 
				-0.02640338f, -0.008124991f, -0.017541144f, -0.007932348f, -0.002098541f, 
				-0.01008603f, -0.025077097f, -0.053949945f, -0.023350062f, -0.029177789f, 
				-0.006354479f, -0.002867078f, -0.004968034f, 0.0039706486f, -0.0015980941f, 
				-0.008641919f, 0.009095047f, 0.003664407f, 0.010882391f, -0.005523261f};
		
		float[] expDelta = new float[]{
				-0.0818294f, -0.16828917f, -0.17053378f, 0.0759145f, 0.13912615f, 0.0329047f, -0.08604437f, -0.19569962f, -0.017746381f, -0.07223136f, -0.20802501f, 0.026594259f, 0.062725715f, -0.14274885f, -0.08169465f, -0.06221814f, -0.0925414f, 0.028002985f, -0.17292416f, -0.17514737f, -0.12002451f, 0.013043992f, -6.0931593E-4f, 0.012378298f, 0.0014778227f, -0.186651f, -0.054080896f, 0.059774034f, -0.22190878f, 0.06828219f, 0.041177664f, -0.23092169f, -0.08979754f, -0.06185287f, -0.14252709f, -0.13840382f, 0.3482734f, 0.35738897f};

		
		float[] o = output.get();
		for (int i = 0; i < expOutput.length; i++) {
			Assert.assertEquals(expOutput[i], o[i], 0.001f);
		}
		
		float[] gi = gradInput.get();
		for (int i = 0; i < expGradInput.length; i++) {
			Assert.assertEquals(expGradInput[i], gi[i], 0.001f);
		}
		
		float[] d = conv.getDeltaParameters().get();
		for(int i=0;i<expDelta.length;i++){
			Assert.assertEquals(expDelta[i], d[i], 0.001f);
		}
		
	}
	
	@Test
	public void testSpatialMaxPooling() throws Exception {
		
		SpatialMaxPooling pool = new SpatialMaxPooling(factory, 2, 2, 2, 2); 
		
		float[] inputData = new float[]{
				-1.0016336f, -0.3659996f, -0.561763f, 0.025022956f, 0.39551026f, 0.25571227f, 
				-0.36720142f, -1.1175866f, 1.6601679f, 0.7697076f, 0.94416225f, -0.33390528f, 
				-0.7185745f, -0.82316816f, -0.4198505f, 0.8720639f, -0.91903704f, 1.1918385f, 
				0.9839494f, -1.1864451f, -0.2240562f, 0.1604332f, -1.2869335f, -0.3002753f, 
				0.70621985f, 1.6712323f, 0.427771f, -1.1304947f, 1.5576284f, 0.8286627f, 
				1.1835989f, -1.261588f, 0.109402984f, 0.31725317f, 1.2378355f, 0.40925455f, 

				-0.56402963f, -1.0522915f, -0.9422165f, 1.0155184f, -0.31722248f, 1.1515416f, 
				-0.14420497f, -1.6517973f, 0.4656332f, 1.050356f, -0.5541778f, -0.21497928f, 
				-0.2700798f, 0.7216463f, 0.10998042f, -0.21850033f, 0.58419466f, 0.59096277f, 
				-1.3353262f, 1.5876176f, -0.9639381f, 0.21641004f, 2.2216365f, -0.19380932f, 
				-0.027697943f, 0.82068586f, 0.7767722f, 0.69155896f, -1.9302452f, 0.16983992f, 
				1.1714138f, -1.4150684f, -0.7196508f, 0.74438053f, 0.88626957f, -1.6724747f
		};
		
		Tensor input = factory.createTensor(inputData, 2,6,6);
		
		float[] gradOutputData = new float[]{
				-0.024800602f, 0.022750217f, 0.22541349f, 
				-0.055356164f, -0.07583931f, 0.10012579f, 
				-0.0026466453f, 0.22068158f, 0.04035959f, 

				0.14729609f, -0.039030015f, -0.017672129f, 
				-0.007538767f, 0.22331862f, -0.076386474f, 
				0.09946432f, -0.021048022f, 0.001844853f
		};
		
		Tensor gradOutput = factory.createTensor(gradOutputData, 2,3,3);
		
		final Tensor output = factory.createTensor(2,3,3);
		final Tensor gradInput = factory.createTensor(2,6,6);
		
		pool.addForwardListener(new ForwardListener() {
			@Override
			public void onForward(UUID moduleId, Tensor o, String... tags) {
				o.copyInto(output);
//				System.out.println("OUTPUT CONV "+output);
				pool.backward(UUID.randomUUID(), gradOutput);
			}
		});
		
		pool.addBackwardListener(new BackwardListener() {
			@Override
			public void onBackward(UUID moduleId, Tensor gi, String... tags) {
				gi.copyInto(gradInput);
//				System.out.println("BACKWARD CONV "+gradInput);
			}
		});
		pool.forward(UUID.randomUUID(), input);

		Thread.sleep(200);
		
		
		float[] expOutput = new float[]{-0.3659996f, 1.6601679f, 0.94416225f, 
				0.9839494f, 0.8720639f, 1.1918385f, 
				1.6712323f, 0.427771f, 1.5576284f, 

				-0.14420497f, 1.050356f, 1.1515416f, 
				1.5876176f, 0.21641004f, 2.2216365f, 
				1.1714138f, 0.7767722f, 0.88626957f};
		
		float[] expGradInput = new float[]{0.0f, -0.024800602f, 0.0f, 0.0f, 0.0f, 0.0f, 
				0.0f, 0.0f, 0.022750217f, 0.0f, 0.22541349f, 0.0f, 
				0.0f, 0.0f, 0.0f, -0.07583931f, 0.0f, 0.10012579f, 
				-0.055356164f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 
				0.0f, -0.0026466453f, 0.22068158f, 0.0f, 0.04035959f, 0.0f, 
				0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 

				0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.017672129f, 
				0.14729609f, 0.0f, 0.0f, -0.039030015f, 0.0f, 0.0f, 
				0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 
				0.0f, -0.007538767f, 0.0f, 0.22331862f, -0.076386474f, 0.0f, 
				0.0f, 0.0f, -0.021048022f, 0.0f, 0.0f, 0.0f, 
				0.09946432f, 0.0f, 0.0f, 0.0f, 0.001844853f, 0.0f};
		
		float[] o = output.get();
		for (int i = 0; i < expOutput.length; i++) {
			Assert.assertEquals(expOutput[i], o[i], 0.001f);
		}
		
		float[] gi = gradInput.get();
		for (int i = 0; i < expGradInput.length; i++) {
			Assert.assertEquals(expGradInput[i], gi[i], 0.001f);
		}
	}
	
//	@Test
//	public void testSpatialConvolutionLena() throws Exception {
//		
//		Tensor input = converter.readFromFile("test/lena.png");
//		
//		//converter.writeToFile("test/r.png", input.select(0, 0));
//		//converter.writeToFile("test/g.png", input.select(0, 1));
//		//converter.writeToFile("test/b.png", input.select(0, 2));
//
//		int noInputPlanes = 3;
//		int noOutputPlanes = 5;
//		int kernelWidth = 3;
//		int kernelHeight = 3;
//		SpatialConvolution conv = new SpatialConvolution(factory, 
//				noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, 1, 1, false);
//		
//		for(int i=0;i<noOutputPlanes;i++){
//			Tensor sub = conv.weights.select(0, i);
//			for(int j=0;j<noInputPlanes;j++){
//				Tensor kernel = sub.select(0, j);
//				// TODO kernel for each outputplane
//				switch(i){
//				case 0:
//					kernel.fill(0.0f);
//					kernel.set(-1.0f/3.0f, 1, 0);
//					kernel.set(2.0f/3.0f, 1, 1);
//					kernel.set(-1.0f/3.0f, 1, 2);
//					break;
//				case 1:
//					kernel.fill(0.0f);
//					kernel.set(-1.0f/3.0f, 0, 1);
//					kernel.set(2.0f/3.0f, 1, 1);
//					kernel.set(-1.0f/3.0f, 2, 1);
//					break;
//				case 2:
//					kernel.fill(-1.0f/3.0f);
//					kernel.set(3.0f, 1, 1);		
//					break;
//				case 3:
//					kernel.fill(0.0f);
//					kernel.set(-1.0f/3.0f, 0, 1);
//					kernel.set(-1.0f/3.0f, 1, 0);
//					kernel.set(-1.0f/3.0f, 0, 0);
//					kernel.set(1.0f/3.0f, 2, 2);
//					kernel.set(1.0f/3.0f, 2, 1);
//					kernel.set(1.0f/3.0f, 1, 2);
//					break;					
//				case 4:
//					kernel.fill(1.0f/3.0f);
//					break;
//				}
//			}
//		}
//		
////		System.out.println(conv.getParameters());
//	
//		Object lock = new Object();
//		conv.addForwardListener(new ForwardListener() {
//			
//			@Override
//			public void onForward(Tensor output, String... tags) {
//				for(int i=0;i<noOutputPlanes;i++){
//					try {
//						//converter.writeToFile("test/output-"+i+".png", output.select(0, i));
//					} catch (Exception e) {
//						e.printStackTrace();
//					}
//				}
//				synchronized(lock){
//					lock.notifyAll();
//				}
//			}
//		});
//		long t1 = System.currentTimeMillis();
//		conv.forward(UUID.randomUUID(), input);
//		synchronized(lock){
//			lock.wait();
//		}
//		long t2 = System.currentTimeMillis();
//		System.out.println("Time "+(t2-t1)+" ms");
//	}
}
