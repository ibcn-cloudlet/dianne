package be.iminds.iot.dianne.nn.module.layer;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.UUID;

import javax.imageio.ImageIO;

import junit.framework.Assert;

import org.junit.Test;

import be.iminds.iot.dianne.nn.module.BackwardListener;
import be.iminds.iot.dianne.nn.module.ForwardListener;
import be.iminds.iot.dianne.nn.module.layer.SpatialConvolution;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.impl.java.JavaTensorFactory;

public class SpatialConvolutionTest {

	private JavaTensorFactory factory = new JavaTensorFactory();
	
	@Test
	public void testSpatialConvolutionConstructor1() {
		int noInputPlanes = 3;
		int noOutputPlanes = 2;
		int kernelWidth = 3;
		int kernelHeight = 3;
		
		SpatialConvolution conv = new SpatialConvolution(factory, noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight);
	
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
		
		SpatialConvolution conv = new SpatialConvolution(factory, noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight);

		for(int i=0;i<noOutputPlanes;i++){
			Tensor sub1 = conv.weights.select(0, i);
			for(int j=0;j<noInputPlanes;j++){
				Tensor sub2 = sub1.select(0, j);
				System.out.println("Kernel:");
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
				System.out.println(sub2);
				System.out.println("===");
			}
		}
		
		conv.bias.set(0.1f, 0);
		conv.bias.set(0.1f, 1);
		
		System.out.println("PARAMS: "+conv.getParameters());

		Tensor input = factory.createTensor(5,5);
		float k = 0;
		for(int i=0;i<5;i++){
			for(int j=0;j<5;j++){
				input.set(k++, i,j);
			}
		}
		System.out.println("INPUT ");
		System.out.println(input);
		
		final Tensor output = factory.createTensor(2,3,3);
		final Tensor gradInput = factory.createTensor(5,5);
		
		conv.addForwardListener(new ForwardListener() {
			@Override
			public void onForward(Tensor o) {
				o.copyInto(output);
				System.out.println("OUTPUT CONV "+output);
				o.fill(0.1f);
				conv.backward(UUID.randomUUID(), o);
			}
		});
		
		conv.addBackwardListener(new BackwardListener() {
			@Override
			public void onBackward(Tensor gi) {
				gi.copyInto(gradInput);
				System.out.println("BACKWARD CONV "+gradInput);
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
		
		Tensor input = readImage("lena.png");

		writeImage(input.select(0, 0), "r.png");
		writeImage(input.select(0, 1), "g.png");
		writeImage(input.select(0, 2), "b.png");

		int noInputPlanes = 3;
		int noOutputPlanes = 5;
		int kernelWidth = 3;
		int kernelHeight = 3;
		SpatialConvolution conv = new SpatialConvolution(factory, 
				noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight);
		
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
		
		System.out.println(conv.getParameters());
	
		Object lock = new Object();
		conv.addForwardListener(new ForwardListener() {
			
			@Override
			public void onForward(Tensor output) {
				for(int i=0;i<noOutputPlanes;i++){
					try {
						writeImage(output.select(0, i), "output-"+i+".png");
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
	
	private Tensor readImage(String file) throws Exception {
		BufferedImage img = ImageIO.read(new File("test/"+file));

		Tensor result = factory.createTensor(3, img.getWidth(), img.getHeight());
		int[] pixels = img.getRGB(0, 0, img.getWidth(), img.getHeight(), null, 0, img.getWidth());

		int k = 0;
		for(int i=0;i<img.getWidth();i++){
			for(int j=0;j<img.getHeight();j++){
				int pixel = pixels[k++];
				int alpha = (pixel >> 24) & 0xff;
			    int red = (pixel >> 16) & 0xff;
			    int green = (pixel >> 8) & 0xff;
			    int blue = (pixel) & 0xff;
	
			    result.set((float)red, 0, i, j);
			    result.set((float)green, 1, i, j);
			    result.set((float)blue, 2, i, j);
			}
		}
		return result;
	}
	
	private void writeImage(Tensor mat, String file) throws Exception {
		int width = mat.dims()[0];
		int height = mat.dims()[1];
		BufferedImage frame = new BufferedImage(width, height,
				BufferedImage.TYPE_INT_RGB);
		float[] data = mat.get();
		int k = 0;
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				int val = (int)data[k++];
				final int r = val;
				final int g = val;
				final int b = val;
				final int a = 255;
				final int col = a << 24 | r << 16 | g << 8 | b;
				frame.setRGB(i, j, col);
			}
		}
		ImageIO.write(frame, "png", new File("test/"+file));
	}
}
