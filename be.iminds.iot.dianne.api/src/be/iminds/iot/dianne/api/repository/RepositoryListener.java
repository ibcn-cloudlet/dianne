package be.iminds.iot.dianne.api.repository;

import java.util.UUID;

public interface RepositoryListener {

	public void onParametersUpdate(UUID moduleId, String... tag);
	
}
