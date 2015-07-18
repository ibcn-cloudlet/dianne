package be.iminds.iot.dianne.api.nn.module;

import java.util.UUID;

import be.iminds.iot.dianne.tensor.Tensor;

public interface Module {
	
	public enum Mode {FORWARD_ON_CHANGE, WAIT_FOR_ALL}
	
	public UUID getId();

	public void forward(final UUID moduleId, final Tensor input, final String... tags);
	
	public void backward(final UUID moduleId, final Tensor gradOutput, final String... tags);
	
	public void setNext(final Module... next);
	
	public void setPrevious(final Module... prev);

	public void addForwardListener(ForwardListener listener);
	
	public void removeForwardListener(ForwardListener listener);
	
	public void addBackwardListener(BackwardListener listener);
	
	public void removeBackwardListener(BackwardListener listener);
	
	public void setMode(Mode mode);
}
