package be.iminds.iot.dianne.api.repository;

import java.util.Collection;
import java.util.UUID;

/**
 * Listener to get notifications when the Repository has updates for a given moduleId.
 * 
 * Listeners are registered with a String[] targets service property, indicating a number of
 * moduleId:tag pairs that identify the moduleId/tags this listener is interested in
 * 
 * @author tverbele
 *
 */
public interface RepositoryListener {

	/**
	 * Notify listener of a parameter update for a collection of moduleIds and optional tags 
	 * @param moduleIds moduleIds whos parameters were updated
	 * @param tag optional tags
	 */
	public void onParametersUpdate(Collection<UUID> moduleIds, String... tag);
	
}
