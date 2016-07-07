package be.iminds.iot.dianne.api.dataset;

import java.util.Map;

import org.osgi.service.component.annotations.Activate;

import be.iminds.iot.dianne.tensor.Tensor;

public abstract class AbstractDataset implements Dataset {
	
	protected String name;
	protected int[] inputDims;
	protected int inputSize;
	protected int[] targetDims;
	protected int targetSize;
	protected int noSamples;
	protected String[] labels;
	protected String labelsFile;
	protected String dir;
	
	@Activate
	protected void activate(Map<String, Object> properties) {
		init(properties);
		
		String d = (String)properties.get("dir");
		if(d != null){
			dir = d;
		}

		if(properties.containsKey("name"))
			this.name = (String)properties.get("name");

		String[] id = (String[])properties.get("inputDims");
		if(id!=null){
			inputDims= new int[id.length];
			for(int i=0;i<id.length;i++){
				inputDims[i] = Integer.parseInt(id[i]);
			}
		}
		
		String[] od = (String[])properties.get("targetDims");
		if(od != null){
			targetDims= new int[od.length];
			for(int i=0;i<od.length;i++){
				targetDims[i] = Integer.parseInt(od[i]);
			}
		}
		
		String ns = (String)properties.get("noSamples");
		if(ns != null)
			noSamples = Integer.parseInt(ns);
		
		if(properties.containsKey("labels")){
			labels = (String[])properties.get("labels");
		} else if(properties.containsKey("labelsFile")){
			labelsFile = (String) properties.get("labelsFile");
		}
		
		if(inputDims != null){
			inputSize = 1;
			for(int i=0;i<inputDims.length;i++){
				inputSize *= inputDims[i];
			}
		} else {
			inputSize = -1;
		}
		
		if(targetDims != null){
			targetSize = 1;
			for(int i=0;i<targetDims.length;i++){
				targetSize *= targetDims[i];
			}
		} else {
			targetSize = -1;
		}
		
		if(labelsFile != null)
			readLabels(labelsFile);
		
	}
	
	protected abstract void init(Map<String, Object> properties);
	
	protected abstract void readLabels(String labelsFile);

	protected abstract Tensor getInputSample(Tensor t, int index);

	protected abstract Tensor getTargetSample(Tensor t, int index);
	
	@Override
	public Sample getSample(Sample s, final int index){
		if(s == null){
			s = new Sample();
		}
		
		s.input = getInputSample(s.input, index);
		s.target = getTargetSample(s.target, index);
		
		return s;
	}
	
	@Override
	public String getName(){
		return name;
	}
	
	@Override
	public int[] inputDims(){
		return inputDims;
	}
	
	@Override
	public int[] targetDims(){
		return targetDims;
	}
	
	@Override
	public int size() {
		return noSamples;
	}

	@Override
	public String[] getLabels(){
		return labels;
	}
	
}