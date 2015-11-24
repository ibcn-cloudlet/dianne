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


public class OverfeatTest extends AbstractDianneTest {

	public final int TEST_SAMPLE = 2;

	private Dataset imagenet;
	
	public void setUp() throws Exception {
    	super.setUp();
    	
    	ServiceReference[] rds = context.getAllServiceReferences(Dataset.class.getName(), null);
    	for(ServiceReference rd : rds){
    		Dataset d = (Dataset) context.getService(rd);
    		if(d.getName().equals("ImageNet")){
    			imagenet = d;
    		}
    	}
    }
	
	public void testOverfeat() throws Exception {
		deployNN("../tools/nn/overfeat_fast_2/modules.txt");
		
		final Tensor sample = imagenet.getInputSample(TEST_SAMPLE);		
		final Tensor result = factory.createTensor(1000);
		
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
				e.printStackTrace();
			}
		});

		// Write intermediate output to file
		for(Module m : getModules()){
			m.addForwardListener(new ForwardListener() {
				@Override
				public void onForward(UUID moduleId, Tensor output, String... tags) {
					try {
						File f = new File("out_"+m.getId()+".txt");
						PrintWriter writer = new PrintWriter(f);
						writer.println(Arrays.toString(output.dims()));
						
						float[] data = output.get();
						for(int i=0;i<data.length;i++){
							writer.write(data[i]+" ");
						}
						writer.close();
					} catch(Exception e){
					}
				}

				@Override
				public void onError(UUID moduleId, ModuleException e, String... tags) {
					e.printStackTrace();
				}
			});
		}
		
		synchronized(lock){
			getInput().input(sample);
			lock.wait();
		}
		
		int index = factory.getTensorMath().argmax(result);
		System.out.println(getOutput().getOutputLabels()[index]+" "+result.get(index));
		
		int expected = factory.getTensorMath().argmax(imagenet.getOutputSample(TEST_SAMPLE));
		System.out.println("Expected: "+getOutput().getOutputLabels()[expected]);
		Assert.assertEquals(expected, index);
	}
}
