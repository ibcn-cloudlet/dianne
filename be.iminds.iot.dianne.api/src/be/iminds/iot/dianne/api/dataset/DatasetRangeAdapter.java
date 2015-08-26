package be.iminds.iot.dianne.api.dataset;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * This Dataset adapter allows you to wrap a Dataset in a new Dataset with only
 * a subset of the samples. Can for example be used to split up a Dataset into
 * a training, validation and test set.
 * 
 * @author tverbele
 *
 */
public class DatasetRangeAdapter implements Dataset {

	private Dataset data;
	
	// start and end index
	private int start;
	private int end;
	
	/**
	 * Creates a new DatasetRangeAdapter
	 * 
	 * @param data the dataset to wrap
	 * @param start the start index for the new dataset
	 * @param end the end index of the new dataset
	 */
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
