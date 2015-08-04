package be.iminds.iot.dianne.api.io;

import java.util.List;
import java.util.UUID;

// link Input modules to actual inputs
public interface InputManager {

	public List<InputDescription> getAvailableInputs();
	
	public void setInput(UUID inputId, UUID nnId, String input);
	
	public void unsetInput(UUID inputId, UUID nnId, String input);

}
