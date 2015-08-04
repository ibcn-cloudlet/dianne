package be.iminds.iot.dianne.api.io;

import java.util.List;
import java.util.UUID;

// link an output module to an actual output actuator
public interface OutputManager {
	
	public List<OutputDescription> getAvailableOutputs();
	
	public void setOutput(UUID outputId, UUID nnId, String output);
	
	public void unsetOutput(UUID outputId, UUID nnId, String output);
}
