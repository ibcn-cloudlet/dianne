package be.iminds.iot.dianne.api.nn.train.api;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Random;
import java.util.concurrent.CountDownLatch;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.module.BackwardListener;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Module.Mode;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.tensor.Tensor;

// convenience class to process a dataset with a neural network one by one
public abstract class DatasetProcessor {

	protected Input input;
	protected Output output;
	protected Dataset data;
	private boolean backpropagate;
	
	private int index = 0;
	private Tensor next = null;
	private Object lock = new Object();
	private boolean shuffle;
	private ArrayList<Integer> indices = null;
	private Random rand = new Random(1234);
	
	public DatasetProcessor(Input input, Output output, Dataset data, boolean backpropagate, boolean shuffle){
		this.input = input;
		this.output = output;
		this.data = data;
		this.backpropagate = backpropagate;
		
		this.shuffle = shuffle;
		this.indices = new ArrayList<Integer>(data.size());
		for(int i=0;i<data.size();i++){
			this.indices.add(new Integer(i));
		}
	}
	
	/**
	 	This method will start processing the complete dataset and block until done.
	 	You can only process one at a time with same object (this method is synchronized).
	 	If boolean backpropagate is set on construction, next item will be processed after onBackward, else the next
	 	item will be processed onForward, you should yourself initiate the backpropagation in onForward though
	  */
	public synchronized void process(){
		// TODO set all modules to BLOCKING?
		input.setMode(EnumSet.of(Mode.BLOCKING));
		
		if(shuffle){
			Collections.shuffle(indices, rand);
		}
		
		next = data.getInputSample(indices.get(index));
		
		final CountDownLatch latch = new CountDownLatch(data.size());
		
		// add input and output listeners
		BackwardListener inputListener = new BackwardListener() {
			
			@Override
			public void onBackward(final Tensor gradInput, final String... tags) {
				DatasetProcessor.this.onBackward(indices.get(index-1), gradInput);
				
				// if backpropagate, forward next
				if(backpropagate){
					// next
					latch.countDown();
					
					next();
				}
			}
		};
		input.addBackwardListener(inputListener);
		
		ForwardListener outputListener = new ForwardListener() {
			
			@Override
			public void onForward(final Tensor output, final String... tags) {
				// index is already advanced by one so subtract here
				DatasetProcessor.this.onForward(indices.get(index-1), output);
				
				if(!backpropagate){
					// next
					latch.countDown();
					
					next();
				}
			}
		};
		output.addForwardListener(outputListener);
		
		// forward first item and load next
		next();
		
		// wait
		try {
			latch.await();
		} catch (InterruptedException e) {
		}
		
		
		// remove listeners again
		input.removeBackwardListener(inputListener);
		output.removeForwardListener(outputListener);
		// reset index
		index = 0;
	}
	
	// forward next sample and load the one after
	private void next(){
		// synchronize in case fetching next takes longer then execution?
		synchronized(lock){
			if(next!=null)
				input.input(next);

			index++;
			if(index < data.size()){
				next = data.getInputSample(indices.get(index));
			} else {
				next = null;
			}
		}
	}
	
	/**
	 * Implement what to do onForward for item on index, if you want to backpropagate, make
	 * sure you call backpropagate on the Output module in onForward
	 */
	protected abstract void onForward(int index, Tensor out);
	
	/**
	 * Implement what to do on onBackward. If you set backpropagate to false, this can be left empty
	 */
	protected abstract void onBackward(int index, Tensor in);
}
