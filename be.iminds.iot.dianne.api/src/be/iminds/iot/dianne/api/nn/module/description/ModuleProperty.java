package be.iminds.iot.dianne.api.nn.module.description;

public class ModuleProperty {

	private final String name;
	private final String id;
	
	public ModuleProperty(String name, String id){
		this.name = name;
		this.id = id;
	}

	public String getName() {
		return name;
	}

	public String getId() {
		return id;
	}

	@Override
	public int hashCode() {
		return id.hashCode();
	}
	
	
}
