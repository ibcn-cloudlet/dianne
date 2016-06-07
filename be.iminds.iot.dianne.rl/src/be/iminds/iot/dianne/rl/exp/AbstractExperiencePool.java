package be.iminds.iot.dianne.rl.exp;

import java.util.Collection;
import java.util.Map;

import be.iminds.iot.dianne.api.dataset.AbstractDataset;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.tensor.Tensor;

public abstract class AbstractExperiencePool extends AbstractDataset implements ExperiencePool {

	protected int maxSize = 1000; // max size of the experience pool
	
	protected int[] stateDims;
	protected int stateSize;
	protected int[] actionDims;
	protected int actionSize;
	protected int sampleSize;
	
	protected float[] emptyState;
	
	private int start = 0; // current start index
	
	@Override
	public void activate(Map<String, Object> config) {
		super.activate(config);
		
		stateDims = inputDims;
		stateSize = inputSize;
	
		actionDims = outputDims;
		actionSize = outputSize;
	
		sampleSize = 2*stateSize+actionSize+1;

		emptyState = new float[stateSize];
		for(int i=0;i<stateSize;i++){
			emptyState[i] = -Float.MAX_VALUE;
		}
		
		setup();
	}
	
	@Override
	protected void init(Map<String, Object> properties) {
		if(properties.containsKey("maxSize"))
			this.maxSize = Integer.parseInt((String) properties.get("maxSize"));
		
		String[] id = (String[])properties.get("stateDims");
		if(id!=null){
			inputDims= new int[id.length];
			for(int i=0;i<id.length;i++){
				inputDims[i] = Integer.parseInt(id[i]);
			}
		}
		
		String[] od = (String[])properties.get("actionDims");
		if(od!=null){
			outputDims= new int[od.length];
			for(int i=0;i<od.length;i++){
				outputDims[i] = Integer.parseInt(od[i]);
			}
		}
	}

	@Override
	protected void readLabels(String labelsFile) {}
	
	@Override
	public Tensor getInputSample(Tensor t, int index) {
		return getState(t, index);
	}

	@Override
	public Tensor getState(Tensor t, int index) {
		float[] sampleBuffer = loadData(getIndex(index));
		float[] stateBuffer = new float[stateSize];
		System.arraycopy(sampleBuffer, 0, stateBuffer, 0, stateSize);

		if(t == null){
			t = new Tensor(stateBuffer, stateDims);
		} else {
			t.set(stateBuffer);
		}
		
		return t;
	}

	@Override
	public Tensor getOutputSample(Tensor t, int index) {
		return getAction(t, index);	
	}

	@Override
	public Tensor getAction(Tensor t, int index) {
		float[] sampleBuffer = loadData(getIndex(index));
		float[] actionBuffer = new float[actionSize];
		System.arraycopy(sampleBuffer, stateSize, actionBuffer, 0, actionSize);
		
		if(t == null){
			t = new Tensor(actionBuffer, actionDims);
		} else {
			t.set(actionBuffer);
		}
		
		return t;
	}

	@Override
	public float getReward(int index) {
		float[] sampleBuffer = loadData(getIndex(index));
		return sampleBuffer[stateSize+actionSize];
	}

	@Override
	public Tensor getNextState(Tensor t, int index) {
		float[] sampleBuffer = loadData(getIndex(index));
		float[] nextStateBuffer = new float[stateSize];
		System.arraycopy(sampleBuffer, stateSize+actionSize+1, nextStateBuffer, 0, stateSize);
		
		if(nextStateBuffer[0] != -Float.MAX_VALUE){
			return null;
		}
		
		if(t == null){
			t = new Tensor(nextStateBuffer, stateDims);
		} else {
			t.set(nextStateBuffer);
		}
		
		return t;
	}
	
	public ExperiencePoolSample getSample(int index) {
		return getSample(null, index);
	}
	
	public ExperiencePoolSample getSample(ExperiencePoolSample s, int index){
		int i = getIndex(index); // TODO does it matter that we shift the index here?!
		
		float[] sampleBuffer = loadData(i);
		float[] stateBuffer = new float[stateSize];
		float[] actionBuffer = new float[actionSize];
		float[] nextStateBuffer = new float[stateSize];
		System.arraycopy(sampleBuffer, 0, stateBuffer, 0, stateSize);
		System.arraycopy(sampleBuffer, stateSize, actionBuffer, 0, actionSize);
		float rewardBuffer = sampleBuffer[stateSize+actionSize];
		System.arraycopy(sampleBuffer, stateSize+actionSize+1, nextStateBuffer, 0, stateSize);
		
		if(s == null){
			s = new ExperiencePoolSample(null, null, 0.0f, null);	
		}
		
		// copy into the existing sample
		if(s.state == null){
			s.state = new Tensor(stateBuffer, stateDims);
		} else {
			s.state.set(stateBuffer);
		}
		
		if(s.action == null){
			s.action = new Tensor(actionBuffer, actionDims);
		} else {
			s.action.set(actionBuffer);
		}
		
		s.reward = rewardBuffer;
		
		if(nextStateBuffer[0] != -Float.MAX_VALUE){
			if(s.nextState == null){
				s.nextState = new Tensor(nextStateBuffer, stateDims);
			} else {
				s.nextState.set(nextStateBuffer);
			}
		} else {
			s.nextState = null;
		}

		return s;
	}
	
	private int getIndex(int index){
		int i = start + index;
		if(i >= noSamples){
			i -= noSamples;
		}
		return i;
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
	public void addSample(Tensor state, Tensor action, float reward,
			Tensor nextState) {
		ExperiencePoolSample s = new ExperiencePoolSample(state, action, reward, nextState);
		addSample(s);
	}
	
	@Override
	public void addSample(ExperiencePoolSample s) {
		add(s);
	}
	
	@Override
	public void addSamples(Collection<ExperiencePoolSample> ss) {
		for(ExperiencePoolSample s : ss){
			add(s);
		}
	}
	
	private void add(ExperiencePoolSample s){
		int index;
		if(noSamples==maxSize){
			// cycling
			index = start++;
			if(start==maxSize){
				start = 0;
			}
		} else {
			index = noSamples++;
		}

		float[] writeBuffer = new float[sampleSize];
		System.arraycopy(s.state.get(), 0, writeBuffer, 0, stateSize);
		System.arraycopy(s.action.get(), 0, writeBuffer, stateSize, actionSize);
		writeBuffer[stateSize+actionSize] = s.reward;
		if(s.nextState == null){
			System.arraycopy(emptyState, 0, writeBuffer, stateSize+actionSize+1, stateSize);
		} else {
			System.arraycopy(s.nextState.get(), 0, writeBuffer, stateSize+actionSize+1, stateSize);
		}
		writeData(index, writeBuffer);
	}
	
	@Override
	public void reset() {
		// TODO concurrency?
		noSamples = 0;
		start = 0;
	}
	
	protected abstract void setup();
	
	protected abstract float[] loadData(int index);
	
	protected abstract void writeData(int index, float[] sample);

}
