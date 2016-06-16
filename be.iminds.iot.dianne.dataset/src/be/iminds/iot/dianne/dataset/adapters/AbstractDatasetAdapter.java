package be.iminds.iot.dianne.dataset.adapters;

import java.util.Map;

import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;

public abstract class AbstractDatasetAdapter implements Dataset {

	protected Dataset data;
	protected String name;

	private Sample temp;
	
	@Reference
	void setDataset(Dataset d){
		this.data = d;
	}
	
	@Activate
	void activate(Map<String, Object> properties) {
		this.name = (String)properties.get("name");
		
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
		return data.inputDims();
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
		return null;
	}
}
