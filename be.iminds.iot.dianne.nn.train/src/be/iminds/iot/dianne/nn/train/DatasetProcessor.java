package be.iminds.iot.dianne.nn.train;

import java.util.concurrent.CountDownLatch;

import be.iminds.iot.dianne.nn.module.Input;
import be.iminds.iot.dianne.nn.module.InputListener;
import be.iminds.iot.dianne.nn.module.Output;
import be.iminds.iot.dianne.nn.module.OutputListener;
import be.iminds.iot.dianne.tensor.Tensor;

// convenience class to process a dataset with a neural network one by one
public abstract class DatasetProcessor {

	private int index = 0;
	
	protected Input input;
	protected Output output;
	protected Dataset data;
	private boolean backpropagate;
	
	public DatasetProcessor(Input input, Output output, Dataset data, boolean backpropagate){
		this.input = input;
		this.output = output;
		this.data = data;
		this.backpropagate = backpropagate;
	}
	
	/**
	 	This method will start processing the complete dataset and block until done.
	 	You can only process one at a time with same object (this method is synchronized).
	 	If boolean backpropagate is set on construction, next item will be processed after onBackward, else the next
	 	item will be processed onForward, you should yourself initiate the backpropagation in onForward though
	  */
	public synchronized void process(){
		
		final CountDownLatch latch = new CountDownLatch(data.size());
		
		// add input and output listeners
		InputListener inputListener = new InputListener() {
			
			@Override
			public void onBackward(Tensor gradInput) {
				DatasetProcessor.this.onBackward(index, gradInput);
				
				// if backpropagate, forward next
				if(backpropagate){
					// next
					latch.countDown();
					
					index++;
					if(index < data.size()){
						Tensor in = data.getInputSample(index);
						input.input(in);
					} 
				}
			}
		};
		input.addInputListener(inputListener);
		
		OutputListener outputListener = new OutputListener() {
			
			@Override
			public void onForward(Tensor output) {
				DatasetProcessor.this.onForward(index, output);
				
				if(!backpropagate){
					// next
					latch.countDown();
					
					index++;
					if(index < data.size()){
						Tensor in = data.getInputSample(index);
						input.input(in);
					} 
				}
			}
		};
		output.addOutputListener(outputListener);
		
		// forward first item
		Tensor in = data.getInputSample(index);
		input.input(in);
		
		// wait
		try {
			latch.await();
		} catch (InterruptedException e) {
		}
		
		
		// remove listeners again
		input.removeInputListener(inputListener);
		output.removeOutputListener(outputListener);
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
