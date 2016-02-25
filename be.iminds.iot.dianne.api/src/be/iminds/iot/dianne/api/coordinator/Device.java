package be.iminds.iot.dianne.api.coordinator;

import java.util.UUID;

public class Device {

	public final UUID id;
	public final String name;
	public final String arch;
	public final String os;
	
	public String ip;
	
	public boolean learn = false;
	public boolean eval = false;
	public boolean act = false;
	
	public double cpuUsage;
	public double memUsage;
	
	public Device(UUID id, String name, String arch, String os, String ip){
		this.id = id;
		this.name = name;
		this.arch = arch;
		this.os = os;
		this.ip = ip;
	}
	
}
