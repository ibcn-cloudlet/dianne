package be.iminds.iot.dianne.api.io;

import java.util.List;
import java.util.UUID;

// link Input modules to actual inputs
public interface InputManager {

	List<InputDescription> getAvailableInputs();
	
	void setInput(UUID inputId, UUID nnId, String input);
	
	void unsetInput(UUID inputId, UUID nnId, String input);
	
}
