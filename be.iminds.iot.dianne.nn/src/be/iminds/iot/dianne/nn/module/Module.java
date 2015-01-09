package be.iminds.iot.dianne.nn.module;

import java.util.UUID;

import be.iminds.iot.dianne.tensor.Tensor;

public interface Module {
	
	public UUID getId();

	public void forward(final UUID moduleId, final Tensor input);
	
	public void backward(final UUID moduleId, final Tensor gradOutput);
	
	public void addNext(final Module... next);
	
	public void removeNext(final Module... next);
	
	public void addPrevious(final Module... prev);
	
	public void removePrevious(final Module... prev);
	
}
