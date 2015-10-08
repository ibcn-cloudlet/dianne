package be.iminds.iot.dianne.api.repository;

import java.util.Collection;
import java.util.UUID;

/**
 * Listener to get notifications when the Repository has updates for a given moduleId.
 * 
 * Listeners are registered with a String[] targets service property, indicating a number of
 * moduleId:tag1:tag2 strings that identify the moduleId/tags this listener is interested in.
 * 
 * One of the tags could also be used to match a neural network instance id one is interested in.
 * For example:
 * targets={"uuid1:run",":uuid2:test"}
 * listens to updates with "run" tag of module with uuid uuid1 and to updates of neural network
 * instance with id uuid2 and tagged with "test"
 * 
 * @author tverbele
 *
 */
public interface RepositoryListener {

	/**
	 * Notify listener of a parameter update for a collection of moduleIds and optional tags 
	 * 
	 * @param nnId the nn instance these parameters originate from
	 * @param moduleIds moduleIds whos parameters were updated
	 * @param tag optional tags
	 */
	public void onParametersUpdate(UUID nnId, Collection<UUID> moduleIds, String... tag);
	
}
