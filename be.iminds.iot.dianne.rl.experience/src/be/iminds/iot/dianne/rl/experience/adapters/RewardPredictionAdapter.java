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
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * This adapter adapts the experience of an experiencepool to a supervised prediction problem
 * where the state is used as input to predict the immediate reward.
 * 
 * @author tverbele
 *
 */
@Component(
		service={Dataset.class},
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.adapters.RewardPredictionAdapter")
public class RewardPredictionAdapter implements Dataset {

	protected ExperiencePool pool;
	protected String name;
	protected Map<String, Object> properties;

	private ExperiencePoolSample temp;
	private boolean binarizeReward = false;

	private int inputSize;
	private int actionSize;
	private int[] inputDims;
	private int[] targetDims = new int[]{1};
	
	@Reference
	void setDataset(ExperiencePool p){
		this.pool = p;
	}
	
	@Activate
	void activate(Map<String, Object> properties) {
		this.properties = properties;
		this.name = (String)properties.get("name");
		
		inputSize = 1;
		for(int i : pool.inputDims()){
			inputSize *= i;
		}
		actionSize = 1;
		for(int a : pool.actionDims()){
			actionSize *= a;
		}
		inputDims = new int[]{inputSize+actionSize};
		
		if(properties.containsKey("binarizeReward")){
			binarizeReward = Boolean.parseBoolean((String)properties.get("binarizeReward"));
		}
	}
	
	
	@Override
	public DatasetDTO getDTO(){
		DatasetDTO dto = pool.getDTO();
		
		dto.name = getName();
		dto.inputDims = inputDims;
		dto.targetDims = targetDims;
		dto.targetType = null; 
		dto.size = size();
		dto.labels = null;
		
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
		return null;
	}
	
	@Override
	public int[] targetDims(){
		return targetDims;
	}
	
	@Override
	public String targetType(){
		return null;
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
			s.input = new Tensor(inputDims);
		}
		temp.input.copyInto(s.input.narrow(0, 0, inputSize));
		temp.target.copyInto(s.input.narrow(0, inputSize, actionSize));
		s.target = temp.reward.copyInto(s.target);
		if(binarizeReward){
			if(s.target.get(0) > 0){
				s.target.set(1.0f, 0);
			} else if(s.target.get(0) < 0){
				s.target.set(0.0f, 0);
			}
		} 
		return s;
	};
	
	@Override
	public String[] getLabels(){
		return null; // no labels?!
	}
}