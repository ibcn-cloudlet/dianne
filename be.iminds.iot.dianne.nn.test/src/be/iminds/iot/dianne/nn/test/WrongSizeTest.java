package be.iminds.iot.dianne.nn.test;

import java.io.File;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.UUID;

import junit.framework.Assert;

import org.osgi.framework.ServiceReference;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.ModuleException;
import be.iminds.iot.dianne.tensor.Tensor;


public class WrongSizeTest extends AbstractDianneTest {

	private Dataset mnist;
	
	public void setUp() throws Exception {
    	super.setUp();
    	
    	ServiceReference[] rds = context.getAllServiceReferences(Dataset.class.getName(), null);
    	for(ServiceReference rd : rds){
    		Dataset d = (Dataset) context.getService(rd);
    		if(d.getName().equals("MNIST")){
    			mnist = d;
    		}
    	}
    }
	
	public void testWrongInputSize() throws Exception {
		deployNN("../tools/nn/mnist-20/modules.txt");
		
		final Tensor sample = factory.createTensor(100);
		final Tensor result = factory.createTensor(10);
		result.fill(0.0f);
	
		
		// wait for output
		final Object lock = new Object();
		getOutput().addForwardListener(new ForwardListener() {
			
			@Override
			public void onForward(UUID moduleId, Tensor output, String... tags) {
				output.copyInto(result);
			
				synchronized(lock){
					lock.notifyAll();
				}
			}

			@Override
			public void onError(UUID moduleId, ModuleException e, String... tags) {
				System.out.println("ERROR!");
				e.printStackTrace();
				
				synchronized(lock){
					lock.notifyAll();
				}
			}
		});
		
//		// Write intermediate output to file
//		for(Module m : getModules()){
//			m.addForwardListener(new ForwardListener() {
//				@Override
//				public void onForward(UUID moduleId, Tensor output, String... tags) {
//					try {
//						File f = new File("out_"+m.getId()+".txt");
//						PrintWriter writer = new PrintWriter(f);
//						writer.println(Arrays.toString(output.dims()));
//						
//						float[] data = output.get();
//						for(int i=0;i<data.length;i++){
//							writer.write(data[i]+" ");
//						}
//						writer.close();
//					} catch(Exception e){
//					}
//				}
//			});
//		}
		
		synchronized(lock){
			getInput().input(sample);
			lock.wait(1000);
		}
		
		Assert.assertTrue(factory.getTensorMath().max(result)==0.0f);
		
		synchronized(lock){
			getInput().input(mnist.getInputSample(0));
			lock.wait(1000);
		}
		
		Assert.assertFalse(factory.getTensorMath().max(result)==0.0f);
	}
}
