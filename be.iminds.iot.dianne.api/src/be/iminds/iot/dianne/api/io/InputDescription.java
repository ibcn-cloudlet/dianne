package be.iminds.iot.dianne.api.io;

public class InputDescription {

	public final String name;
	public final String type;
	
	public InputDescription(String name, String type){
		this.name = name;
		this.type = type;
	}
	
	public String getName(){
		return name;
	}
	
	public String getType(){
		return type;
	}
}
