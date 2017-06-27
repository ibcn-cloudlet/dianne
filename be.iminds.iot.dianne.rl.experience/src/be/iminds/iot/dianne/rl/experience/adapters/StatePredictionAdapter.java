package be.iminds.iot.dianne.rl.experience.adapters;

import java.lang.reflect.Field;
import java.util.Map;

import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.DatasetDTO;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;

/**
 * This adapter adapts the state of an experiencepool to a supervised prediction problem
 * where one part of the state is used as input to predict the other part of the state.
 * 
 * @author tverbele
 *
 */
@Component(
		service={Dataset.class},
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.adapters.StatePredictionAdapter")
public class StatePredictionAdapter implements Dataset {

	protected ExperiencePool pool;
	protected String name;
	protected Map<String, Object> properties;

	private ExperiencePoolSample temp;
	
	private int[] inputNarrow;
	private int[] targetNarrow;
 	
	private int[] inputDims;
	private String inputType;
	private int[] targetDims;
	private String targetType;
	
	@Reference
	void setDataset(ExperiencePool p){
		this.pool = p;
	}
	
	@Activate
	void activate(Map<String, Object> properties) {
		this.properties = properties;
		this.name = (String)properties.get("name");

		// how to narrow the state for the input
		String[] d = (String[])properties.get("inputNarrow");
		inputNarrow = new int[d.length];
		inputDims = new int[d.length/2];
		for(int i=0;i<d.length;i+=2){
			inputNarrow[i] = Integer.parseInt(d[i].trim());
			inputNarrow[i+1] = Integer.parseInt(d[i+1].trim());
			inputDims[i/2] = inputNarrow[i+1]-inputNarrow[i];
		}
		
		inputType = (String)properties.get("inputType");
		
		// how to narrow the state for the target
		d = (String[])properties.get("targetNarrow");
		targetNarrow = new int[d.length];
		targetDims = new int[d.length/2];
		for(int i=0;i<d.length;i+=2){
			targetNarrow[i] = Integer.parseInt(d[i].trim());
			targetNarrow[i+1] = Integer.parseInt(d[i+1].trim());
			targetDims[i/2] = targetNarrow[i+1];
		}
		
		targetType = (String)properties.get("targetType");
	}
	
	
	@Override
	public DatasetDTO getDTO(){
		DatasetDTO dto = pool.getDTO();
		
		dto.name = getName();
		dto.inputDims = inputDims;
		dto.inputType = inputType;
		dto.targetDims = targetDims;
		dto.targetType = targetType; 
		dto.size = size();
		dto.labels = getLabels();
		
		properties.entrySet().forEach(e -> {
			if(e.getKey().contains("."))
				return;
			
			for(Field f : DatasetDTO.class.getFields()){
				if(f.getName().equals(e.getKey()))
					return;
			}
			dto.properties.put(e.getKey(), e.getValue().toString());
		});
		
		return dto;
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
	public String inputType(){
		return inputType;
	}
	
	@Override
	public int[] targetDims(){
		return targetDims;
	}
	
	@Override
	public String targetType(){
		return targetType;
	}
	
	@Override
	public int size() {
		return pool.size();
	}

	@Override
	public synchronized Sample getSample(Sample s, int index) {
		temp = pool.getSample(temp, index);
		if(s == null){
			s = new Sample();
		}
		s.input = temp.input.narrow(inputNarrow).copyInto(s.input);
		s.target = temp.input.narrow(targetNarrow).copyInto(s.target);
		return s;
	};
	
	@Override
	public String[] getLabels(){
		return null; // no labels?!
	}
}