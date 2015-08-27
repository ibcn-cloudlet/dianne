package be.iminds.iot.dianne.api.rl;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * A helper class for representing one sample of an experience pool
 * 
 * @author tverbele
 *
 */
public class ExperiencePoolSample {

	public final Tensor state;
	public final Tensor action;
	public final float reward;
	public final Tensor nextState;
	
	public ExperiencePoolSample(
			final Tensor state, 
			final Tensor action,
			final float reward,
			final Tensor nextState){
		this.state = state;
		this.action = action;
		this.reward = reward;
		this.nextState = nextState;
	}
}
