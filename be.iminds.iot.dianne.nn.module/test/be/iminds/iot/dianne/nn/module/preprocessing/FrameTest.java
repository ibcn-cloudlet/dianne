package be.iminds.iot.dianne.nn.module.preprocessing;

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
public class FrameTest {

	private TensorFactory factory;
	private ImageConverter converter;

	public FrameTest(TensorFactory f, String name) {
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
	public void testFrame() throws Exception {
	
		Frame frame = new Frame(factory, 3, 231, 231);
		
		Tensor input = converter.readFromFile("test/snake.jpg");
		converter.writeToFile("test/out.png", input);

		Object lock = new Object();
		frame.addForwardListener(new ForwardListener() {
			
			@Override
			public void onForward(UUID moduleId, Tensor output, String... tags) {
				System.out.println(Arrays.toString(output.dims()));
				try {
					converter.writeToFile("test/framed-"+factory.getClass().getName()+".png", output);
				} catch (Exception e) {
					e.printStackTrace();
				}
				synchronized(lock){
					lock.notifyAll();
				}
			}

			@Override
			public void onError(UUID moduleId, ModuleException e, String... tags) {
				e.printStackTrace();
			}
		});
		long t1 = System.currentTimeMillis();
		frame.forward(UUID.randomUUID(), input);
		synchronized(lock){
			lock.wait();
		}
		long t2 = System.currentTimeMillis();
		System.out.println("Time "+(t2-t1)+" ms");
	}
}
