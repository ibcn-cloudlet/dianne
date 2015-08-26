package be.iminds.iot.dianne.api.dataset;

import java.util.Arrays;
import java.util.List;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

/**
 * This Dataset adapter allows you to wrap a Dataset in a new Dataset with only
 * a subset of the labels available. The other labels can be either ignored or 
 * be aggregated into one "other" class.
 * 
 * @author tverbele
 *
 */
public class DatasetLabelAdapter implements Dataset {
	
	// required to create new output tensor
	private final TensorFactory factory;
	
	private Dataset data;
	
	// label indices
	private int[] labelIndices;
	private String[] labels;
	// add "other" category
	private boolean other;
	
	/**
	 * Create a new DatasetLabelAdapter
	 * 
	 * @param f tensor factory to use for creating Tensors
	 * @param data the dataset to wrap
	 * @param labels the labels to keep
	 * @param other whether or not the other labels should be aggregated into an "other" class
	 * 
	 */
	public DatasetLabelAdapter(TensorFactory f, Dataset data, String[] labels, boolean other) {
		this.factory = f;
		this.data = data;
		this.other = other;
		this.labels = new String[labels.length+ (other ? 1 : 0)];
		System.arraycopy(labels, 0, this.labels, 0, labels.length);
		if(other){
			this.labels[labels.length] = "other";
		}
		this.labelIndices = new int[labels.length];
		List<String> labelsList = Arrays.asList(data.getLabels());
		for(int i=0;i<labels.length;i++){
			labelIndices[i] = labelsList.indexOf(labels[i]);
		}
	}
	
	@Override
	public String getName(){
		return data.getName();
	}
	
	@Override
	public int size() {
		return data.size();
	}

	@Override
	public Tensor getInputSample(int index) {
		return data.getInputSample(index);
	}

	@Override
	public Tensor getOutputSample(int index) {
		// TODO adapt outputsample
		Tensor t = data.getOutputSample(index);
		Tensor t2 = factory.createTensor(labels.length);
		for(int i=0;i<labelIndices.length;i++){
			t2.set(t.get(labelIndices[i]), i);
		}
		if(other){
			if(factory.getTensorMath().sum(t2)==0){
				t2.set(1.0f, labels.length-1);
			}
		}
		return t2;
	}
	
	@Override
	public String[] getLabels(){
		return labels;
	}

}
