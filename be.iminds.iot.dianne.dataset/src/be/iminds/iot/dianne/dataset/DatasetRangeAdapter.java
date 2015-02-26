package be.iminds.iot.dianne.dataset;

import be.iminds.iot.dianne.tensor.Tensor;

public class DatasetRangeAdapter implements Dataset {

	private Dataset data;
	
	// start and end index
	private int start;
	private int end;
	
	public DatasetRangeAdapter(Dataset data, int start, int end) {
		this.start = start;
		this.end = end;
		this.data = data;
	}
	
	@Override
	public String getName(){
		return data.getName();
	}
	
	@Override
	public int size() {
		return end-start;
	}

	@Override
	public int inputSize() {
		return data.inputSize();
	}

	@Override
	public int outputSize() {
		return data.outputSize();
	}

	@Override
	public Tensor getInputSample(int index) {
		return data.getInputSample(start+index);
	}

	@Override
	public Tensor getOutputSample(int index) {
		return data.getOutputSample(start+index);
	}
	
	@Override
	public String[] getLabels(){
		return data.getLabels();
	}

}
