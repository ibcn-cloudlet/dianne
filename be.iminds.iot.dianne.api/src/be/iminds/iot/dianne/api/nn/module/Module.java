package be.iminds.iot.dianne.api.nn.module;

import java.util.EnumSet;
import java.util.UUID;

import be.iminds.iot.dianne.tensor.Tensor;

public interface Module {
	
	// BLOCKING : block until next is done
	// SKIP : in case next is still busy, skip this frame
	// FORWARD_ON_CHANGE : in case of Fork/Join: forward each time a subset of the input changes
	// WAIT_FOR_ALL : in case of Fork/Join : wait for all inputs to be gathered before forwarding
	public enum Mode {BLOCKING, SKIP, FORWARD_ON_CHANGE, WAIT_FOR_ALL}
	
	UUID getId();

	void forward(final UUID moduleId, final Tensor input, final String... tags);
	
	void backward(final UUID moduleId, final Tensor gradOutput, final String... tags);
	
	void setNext(final Module... next);
	
	void setPrevious(final Module... prev);

	void addForwardListener(ForwardListener listener);
	
	void removeForwardListener(ForwardListener listener);
	
	void addBackwardListener(BackwardListener listener);
	
	void removeBackwardListener(BackwardListener listener);
	
	void setMode(EnumSet<Mode> mode);
}
