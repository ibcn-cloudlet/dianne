package be.iminds.iot.dianne.rl.exp;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Collection;
import java.util.LinkedList;
import java.util.Map;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.api.rl.ExperiencePool;
import be.iminds.iot.dianne.api.rl.ExperiencePoolSample;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(immediate=true,
	configurationPolicy=ConfigurationPolicy.REQUIRE,
	service={ExperiencePool.class, Dataset.class},
	property={"aiolos.unique=true"})
public class FileExperiencePool implements ExperiencePool {

	private TensorFactory factory;

	private String name; // name of the repo
	private String dir; // directory where to store the samples
	private String[] labels; // labels for the actions
	private int stateSize; // size of the state Tensor
	private int actionSize; // size of the action Tensor
	private int maxSize = 1000000; // max size of the experience pool
	
	private ReadWriteLock rwLock = new ReentrantReadWriteLock(true);
	
	private final LinkedList<ExperiencePoolSample> samples = new LinkedList<ExperiencePoolSample>();
	
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
		
		if(config.containsKey("maxSize"))
			this.maxSize = (Integer) config.get("maxSize");

		
		// read from file 
		try {
			file = new File(dir+File.separator+"data.bin");
			
			if(!file.exists())
				file.createNewFile();
			
			input = new DataInputStream(new BufferedInputStream(new FileInputStream(file)));
			output = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file, true)));
		
			rwLock.writeLock().lock();

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
					add(s);
				}
			} catch(IOException e){
				// read till exception end
			} finally {
				rwLock.writeLock().unlock();
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
		add(s);

		
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
	
	@Override
	public void addSamples(Collection<ExperiencePoolSample> ss) {
		rwLock.writeLock().lock();
		for(ExperiencePoolSample s : ss){
			samples.add(s);
		}
		while(samples.size() > maxSize){
			samples.removeFirst();
		}
		rwLock.writeLock().unlock();	
	}
	
	private void add(ExperiencePoolSample s){
		rwLock.writeLock().lock();
		samples.add(s);
		if(samples.size() > maxSize){
			samples.removeFirst();
		}
		rwLock.writeLock().unlock();	
	}
	
	@Reference
	public void setTensorFactory(TensorFactory f){
		this.factory = f;
	}

	@Override
	public void lock() {
		rwLock.readLock().lock();
	}

	@Override
	public void unlock() {
		rwLock.readLock().unlock();
	}

	@Override
	public void reset() {
		rwLock.writeLock().lock();
		
		try {
			samples.clear();

			file.delete();
			file = new File(dir+File.separator+"data.bin");
		} finally {
			rwLock.writeLock().unlock();
		}
	}

}
