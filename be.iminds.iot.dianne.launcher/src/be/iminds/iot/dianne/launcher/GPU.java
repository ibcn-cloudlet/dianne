package be.iminds.iot.dianne.launcher;

import java.util.HashMap;
import java.util.Map;

public class GPU {

	String vendor;
	String model;
	int memory;
	
	Map<String, String> properties = new HashMap<>();
	
	public String toString(){
		StringBuilder b = new StringBuilder();
		b.append(model).append(" (").append(vendor).append(") \tMemory: ").append(memory).append("MB");
		properties.entrySet().forEach(e-> b.append("\n\t- ").append(e.getKey()).append("=").append(e.getValue()));
		return b.toString();
	}
}
