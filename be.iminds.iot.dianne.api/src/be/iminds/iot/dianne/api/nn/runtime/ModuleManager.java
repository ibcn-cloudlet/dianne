package be.iminds.iot.dianne.api.nn.runtime;

import java.util.List;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleTypeDTO;

/**
 * The ModuleManager is responsible for the correct deployment and configuration 
 * of individual Modules. It will call the right factory for creating module instances,
 * and correctly configure the next and previous of each Module.
 * 
 * @author tverbele
 *
 */
public interface ModuleManager {

	/**
	 * The unique identifier of this ModuleManager. Is usually the same as the OSGi framework UUID
	 * 
	 * @return the uuid of this DianneRuntime
	 */
	UUID getRuntimeId();
	
	/**
	 * Deploy a single Module on this runtime
	 * 
	 * @param dto the ModuleDTO describing which Module to deploy
	 * @param nnId the neural network instance this module instance will belong to
	 * @return the ModuleInstanceDTO of the deployed module
	 * @throws InstantiationException when it failed to construct the desired module instance
	 */
	ModuleInstanceDTO deployModule(ModuleDTO dto, UUID nnId) throws InstantiationException;
	
	/**
	 * Undeploy a single ModuleInstance on this runtime
	 * 
	 * @param module the module instance to undeploy
	 */
	void undeployModule(ModuleInstanceDTO module);
	
	/**
	 * Undeploy all ModuleInstances from a given neural network instance that reside on this runtime
	 * 
	 * @param nnId the neural network instance id of the neural network instance to undeploy
	 */
	void undeployModules(UUID nnId);
	
	/**
	 * Get a list of all ModuleInstances that are deployed on this runtime
	 * 
	 * @return list of deployed module instances
	 */
	List<ModuleInstanceDTO> getModules();
	
	/**
	 * Get a list of supported module types that this ModuleManager can deploy. 
	 * 
	 * This is the aggregated list of supported ModuleTypes that the ModuleFactories support
	 * that this ModuleManager as access to.
	 * 
	 * @return list of supported module types
	 */
	List<ModuleTypeDTO> getSupportedModules();
	
	/**
	 * Get a reference to the actual Module service for a given moduleId/nnId
	 * @param moduleId
	 * @param nnId
	 * @return reference to the Module service for this module
	 */
	Module getModule(UUID moduleId, UUID nnId);
	
}
