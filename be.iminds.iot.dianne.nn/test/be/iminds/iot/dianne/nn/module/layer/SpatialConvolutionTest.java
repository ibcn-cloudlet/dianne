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
	
	@Test
	public void testSpatialConvolution() throws InterruptedException {
		int noInputPlanes = 1;
		int noOutputPlanes = 2;
		int kernelWidth = 3;
		int kernelHeight = 3;
		
		SpatialConvolution conv = new SpatialConvolution(factory, noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, 1 ,1, false);

		for(int i=0;i<noOutputPlanes;i++){
			Tensor sub1 = conv.weights.select(0, i);
			for(int j=0;j<noInputPlanes;j++){
				Tensor sub2 = sub1.select(0, j);
//				System.out.println("Kernel:");
				// some self induced kernels
				switch(i){
				case 0:
					sub2.fill(0.0f);
					sub2.set(1.0f, 1, 1);
					break;
				case 1:
					sub2.fill(0.0f);
					sub2.set(-1.0f, 1, 0);
					sub2.set(1.0f, 1, 2);
					break;
				}
//				System.out.println(sub2);
//				System.out.println("===");
			}
		}
		
		conv.bias.set(0.1f, 0);
		conv.bias.set(0.1f, 1);
		
//		System.out.println("PARAMS: "+conv.getParameters());

		Tensor input = factory.createTensor(1,5,5);
		float k = 0;
		for(int i=0;i<5;i++){
			for(int j=0;j<5;j++){
				input.set(k++, 0, i,j);
			}
		}
//		System.out.println("INPUT ");
//		System.out.println(input);
		
		final Tensor output = factory.createTensor(2,3,3);
		final Tensor gradInput = factory.createTensor(1,5,5);
		
		conv.addForwardListener(new ForwardListener() {
			@Override
			public void onForward(Tensor o, String... tags) {
				o.copyInto(output);
//				System.out.println("OUTPUT CONV "+output);
				o.fill(0.1f);
				conv.backward(UUID.randomUUID(), o);
			}
		});
		
		conv.addBackwardListener(new BackwardListener() {
			@Override
			public void onBackward(Tensor gi, String... tags) {
				gi.copyInto(gradInput);
//				System.out.println("BACKWARD CONV "+gradInput);
			}
		});
		conv.forward(UUID.randomUUID(), input);
		
		Thread.sleep(200);
		conv.accGradParameters();
		
		float[] expOutput = new float[]{6.1f, 7.1f, 8.1f, 11.1f, 12.1f, 13.1f, 16.1f, 17.1f,
				18.1f, 2.1f, 2.1f, 2.1f, 2.1f, 2.1f, 2.1f, 2.1f, 2.1f, 2.1f};

		float[] expGradInput = new float[]{0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
				-0.1f, 0.0f, 0.1f, 0.2f, 0.1f, -0.1f, 0.0f, 0.1f, 0.2f, 0.1f,
				-0.1f, 0.0f, 0.1f, 0.2f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
		
		for (int i = 0; i < expOutput.length; i++) {
			Assert.assertEquals(expOutput[i], output.get()[i], 0.001f);
		}
		for (int i = 0; i < expGradInput.length; i++) {
			Assert.assertEquals(expGradInput[i], gradInput.get()[i], 0.001f);
		}
	}
	
	@Test
	public void testSpatialConvolutionLena() throws Exception {
		
		Tensor input = converter.readFromFile("test/lena.png");
		
		//converter.writeToFile("test/r.png", input.select(0, 0));
		//converter.writeToFile("test/g.png", input.select(0, 1));
		//converter.writeToFile("test/b.png", input.select(0, 2));

		int noInputPlanes = 3;
		int noOutputPlanes = 5;
		int kernelWidth = 3;
		int kernelHeight = 3;
		SpatialConvolution conv = new SpatialConvolution(factory, 
				noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, 1, 1, false);
		
		for(int i=0;i<noOutputPlanes;i++){
			Tensor sub = conv.weights.select(0, i);
			for(int j=0;j<noInputPlanes;j++){
				Tensor kernel = sub.select(0, j);
				// TODO kernel for each outputplane
				switch(i){
				case 0:
					kernel.fill(0.0f);
					kernel.set(-1.0f/3.0f, 1, 0);
					kernel.set(2.0f/3.0f, 1, 1);
					kernel.set(-1.0f/3.0f, 1, 2);
					break;
				case 1:
					kernel.fill(0.0f);
					kernel.set(-1.0f/3.0f, 0, 1);
					kernel.set(2.0f/3.0f, 1, 1);
					kernel.set(-1.0f/3.0f, 2, 1);
					break;
				case 2:
					kernel.fill(-1.0f/3.0f);
					kernel.set(3.0f, 1, 1);		
					break;
				case 3:
					kernel.fill(0.0f);
					kernel.set(-1.0f/3.0f, 0, 1);
					kernel.set(-1.0f/3.0f, 1, 0);
					kernel.set(-1.0f/3.0f, 0, 0);
					kernel.set(1.0f/3.0f, 2, 2);
					kernel.set(1.0f/3.0f, 2, 1);
					kernel.set(1.0f/3.0f, 1, 2);
					break;					
				case 4:
					kernel.fill(1.0f/3.0f);
					break;
				}
			}
		}
		
//		System.out.println(conv.getParameters());
	
		Object lock = new Object();
		conv.addForwardListener(new ForwardListener() {
			
			@Override
			public void onForward(Tensor output, String... tags) {
				for(int i=0;i<noOutputPlanes;i++){
					try {
						//converter.writeToFile("test/output-"+i+".png", output.select(0, i));
					} catch (Exception e) {
						e.printStackTrace();
					}
				}
				synchronized(lock){
					lock.notifyAll();
				}
			}
		});
		long t1 = System.currentTimeMillis();
		conv.forward(UUID.randomUUID(), input);
		synchronized(lock){
			lock.wait();
		}
		long t2 = System.currentTimeMillis();
		System.out.println("Time "+(t2-t1)+" ms");
	}
}
