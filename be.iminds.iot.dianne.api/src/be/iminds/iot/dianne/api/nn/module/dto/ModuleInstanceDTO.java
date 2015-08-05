package be.iminds.iot.dianne.api.nn.module.dto;

import java.util.UUID;

/**
 * Represents an actual instance of a Neural Network module
 * 
 * Uniquely identified by the moduleId (which module is it), the runtimeId 
 * (where is it deployed) and nnId (which neural network instance does it belong to)
 * 
 * @author tverbele
 *
 */
public class ModuleInstanceDTO {

	// Module UUID of this module
	public final UUID moduleId;
	
	// UUID of the Neural Network this instance belongs to
	public final UUID nnId;
	
	// UUID of the runtime where the module instance is deployed
	public final UUID runtimeId;

	// Module type
	public final ModuleDTO module;
	
	public ModuleInstanceDTO(ModuleDTO module, UUID nnId, UUID runtimeId){
		this.moduleId = module.id;
		this.nnId = nnId;
		this.runtimeId = runtimeId;
		this.module = module;
	}
	
	@Override
	public boolean equals(Object o){
		if(!(o instanceof ModuleInstanceDTO)){
			return false;
		}
		
		ModuleInstanceDTO other = (ModuleInstanceDTO) o;
		return other.moduleId.equals(moduleId)
				&&	other.nnId.equals(nnId)
				&&  other.runtimeId.equals(runtimeId);
	}
	
	@Override
	public int hashCode(){
		return moduleId.hashCode() + 31*nnId.hashCode() + runtimeId.hashCode();
	}
}
