package be.iminds.iot.dianne.api.coordinator;

import java.util.UUID;

public class Notification {

	public enum Level {
		INFO,
		WARNING,
		DANGER,
		SUCCESS
	}
	
	public final Level level;
	public final String message;
	public final long timestamp;
	
	public Notification(Level level, String message){
		this.level = level;
		this.message = message;
		this.timestamp = System.currentTimeMillis();
	}
}
