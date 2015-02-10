package be.iminds.iot.dianne.nn.module;

import java.util.UUID;

import be.iminds.iot.dianne.tensor.Tensor;

public interface Module {
	
	public UUID getId();

	public void forward(final UUID moduleId, final Tensor input);
	
	public void backward(final UUID moduleId, final Tensor gradOutput);
	
	public void setNext(final Module... next);
	
	public void setPrevious(final Module... prev);

	
	public void addForwardListener(ForwardListener listener);
	
	public void removeForwardListener(ForwardListener listener);
	
	public void addBackwardListener(BackwardListener listener);
	
	public void removeBackwardListener(BackwardListener listener);
	
}
