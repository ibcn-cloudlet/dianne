package be.iminds.iot.dianne.api.io;

import java.util.List;
import java.util.UUID;

// link Input modules to actual inputs
public interface InputManager {

	public List<InputDescription> getAvailableInputs();
	
	public void setInput(UUID inputId, String input);
	
	public void unsetInput(UUID inputId, String input);

}
