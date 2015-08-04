package be.iminds.iot.dianne.api.io;

import java.util.List;
import java.util.UUID;

// link an output module to an actual output actuator
public interface OutputManager {
	
	List<OutputDescription> getAvailableOutputs();
	
	void setOutput(UUID outputId, UUID nnId, String output);
	
	void unsetOutput(UUID outputId, UUID nnId, String output);
}
