package be.iminds.iot.dianne.dataset.adapters;

import java.util.Map;

import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;

public abstract class AbstractDatasetAdapter implements Dataset {

	protected Dataset data;
	protected String name;
	
	protected boolean targetDimsSameAsInput = false;

	private Sample temp;
	
	@Reference
	void setDataset(Dataset d){
		this.data = d;
	}
	
	@Activate
	void activate(Map<String, Object> properties) {
		this.name = (String)properties.get("name");

		// mark if targetDims are same as inputs
		// often requires adapters to also adapt target
		int[] inputDims = data.inputDims();
		if(inputDims != null){
			int[] targetDims = data.targetDims();
			if(inputDims.length == targetDims.length){
				targetDimsSameAsInput = true;
				for(int i=0;i<inputDims.length;i++){
					if(inputDims[i] != targetDims[i]){
						targetDimsSameAsInput = false;
						break;
					}
				}
			}
		}
		
		configure(properties);
	}
	
	protected abstract void configure(Map<String, Object> properties);
	
	@Override
	public String getName(){
		return name;
	}
	
	@Override
	public int[] inputDims(){
		return data.inputDims();
	}
	
	@Override
	public int[] targetDims(){
		return data.targetDims();
	}
	
	@Override
	public int size() {
		return data.size();
	}

	@Override
	public synchronized Sample getSample(Sample s, int index) {
		temp = data.getSample(temp, index);
		if(s == null){
			s = new Sample();
		}
		adaptSample(temp, s);
		return s;
	};
	
	protected abstract void adaptSample(Sample original, Sample adapted);
	
	@Override
	public String[] getLabels(){
		return data.getLabels();
	}
}
