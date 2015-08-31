package be.iminds.iot.dianne.rl.exp;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.rl.ExperiencePool;
import be.iminds.iot.dianne.api.rl.ExperiencePoolSample;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(immediate=true,
	configurationPolicy=ConfigurationPolicy.REQUIRE,
	service={ExperiencePool.class, Dataset.class})
public class FileExperiencePool implements ExperiencePool {

	private TensorFactory factory;

	private String name; // name of the repo
	private String dir; // directory where to store the samples
	private String[] labels; // labels for the actions
	private int stateSize; // size of the state Tensor
	private int actionSize; // size of the action Tensor
	
	
	private final List<ExperiencePoolSample> samples = Collections.synchronizedList(
			new ArrayList<ExperiencePoolSample>());
	
	private File file;
	private DataInputStream input;
	private DataOutputStream output;
	
	@Activate
	public void activate(Map<String, Object> config){
		this.name = (String) config.get("name");
		this.dir = (String) config.get("dir");
		this.labels = (String[]) config.get("labels");
		this.actionSize = (Integer) config.get("actionSize");
		this.stateSize = (Integer) config.get("stateSize");

		// read from file 
		try {
			file = new File(dir+File.separator+"data.bin");
			
			if(!file.exists())
				file.createNewFile();
			
			input = new DataInputStream(new BufferedInputStream(new FileInputStream(file)));
			output = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file, true)));
		
			try {
				while(true){
					float[] stateData = new float[stateSize];
					float[] actionData = new float[actionSize];
					float reward;
					float[] nextStateData = new float[stateSize];
					
					for(int i=0;i<stateSize;i++){
						stateData[i] = input.readFloat();
					}
					
					for(int i=0;i<actionSize;i++){
						actionData[i] = input.readFloat();
					}
					
					reward = input.readFloat();
					
					for(int i=0;i<stateSize;i++){
						nextStateData[i] = input.readFloat();
					}
					
					Tensor state = factory.createTensor(stateData, stateSize);
					Tensor action = factory.createTensor(actionData, actionSize);
					Tensor nextState = factory.createTensor(nextStateData, stateSize);
					
					ExperiencePoolSample s = new ExperiencePoolSample(state, action, reward, nextState);
					samples.add(s);
				}
			} catch(IOException e){
				// read till exception end
			}
			
		} catch(IOException e){
			System.err.println("Failed to read experience pool file for "+name);
		}
	}
	
	@Override
	public int size() {
		return samples.size();
	}
	
	@Override
	public Tensor getInputSample(int index) {
		return samples.get(index).state;
	}

	@Override
	public Tensor getState(int index) {
		return samples.get(index).state;
	}

	@Override
	public Tensor getOutputSample(int index) {
		return samples.get(index).action;
	}

	@Override
	public Tensor getAction(int index) {
		return samples.get(index).action;
	}

	@Override
	public String getName() {
		return name;
	}

	@Override
	public String[] getLabels() {
		return labels;
	}

	@Override
	public float getReward(int index) {
		return samples.get(index).reward;
	}

	@Override
	public Tensor getNextState(int index) {
		return samples.get(index).nextState;
	}

	@Override
	public void addSample(Tensor state, Tensor action, float reward,
			Tensor nextState) {
		ExperiencePoolSample s = new ExperiencePoolSample(state, action, reward, nextState);
		samples.add(s);
		
		// append to file
		synchronized(file){
			try {
				float[] stateData = state.get();
				for(int i=0;i<stateData.length;i++){
					output.writeFloat(stateData[i]);
				}
				
				float[] actionData = action.get();
				for(int i=0;i<actionData.length;i++){
					output.writeFloat(actionData[i]);
				}
				
				output.writeFloat(reward);
				
				float[] nextStateData = nextState.get();
				for(int i=0;i<nextStateData.length;i++){
					output.writeFloat(nextStateData[i]);
				}
			
			} catch(IOException e){
				System.err.println("Failed to write sample to file for experience pool "+name);
			}
		}
	}
	
	@Reference
	public void setTensorFactory(TensorFactory f){
		this.factory = f;
	}
}
