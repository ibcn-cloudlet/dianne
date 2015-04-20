package be.iminds.iot.dianne.nn.test;

import junit.framework.Assert;

import org.osgi.framework.ServiceReference;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.tensor.Tensor;


public class OverfeatTest extends AbstractDianneTest {

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
		deployNN("../tools/nn/overfeat_fast/modules.txt");
		
		final Tensor sample = imagenet.getInputSample(0);
		final Tensor result = factory.createTensor(1000);
		
		// wait for output
		final Object lock = new Object();
		getOutput().addForwardListener(new ForwardListener() {
			
			@Override
			public void onForward(Tensor output) {
				output.copyInto(result);
			
				synchronized(lock){
					lock.notifyAll();
				}
			}
		});
		synchronized(lock){
			getInput().input(sample);
			lock.wait();
		}
		
		int index = factory.getTensorMath().argmax(result);
		System.out.println(getOutput().getOutputLabels()[index]+" "+result.get(index));
		
		int expected = factory.getTensorMath().argmax(imagenet.getOutputSample(0));
		System.out.println("Expected: "+getOutput().getOutputLabels()[expected]);
		Assert.assertEquals(expected, index);
	}
}
