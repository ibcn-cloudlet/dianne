package be.iminds.iot.dianne.api.io;

import java.util.List;
import java.util.UUID;

// link an output module to an actual output actuator
public interface OutputManager {
	
	public List<String> getAvailableOutputs();
	
	public void setOutput(UUID outputId, String output);
	
	public void unsetOutput(UUID outputId, String output);
}
