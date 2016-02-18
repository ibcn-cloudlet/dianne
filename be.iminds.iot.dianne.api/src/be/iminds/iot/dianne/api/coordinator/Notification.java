package be.iminds.iot.dianne.api.coordinator;

import java.util.UUID;

public class Notification {

	public enum Level {
		INFO,
		WARNING,
		DANGER,
		SUCCESS
	}
	
	public final UUID jobId;
	public final Level level;
	public final String message;
	public final long timestamp;
	
	public Notification(UUID jobId, Level level, String message){
		this.jobId = jobId;
		this.level = level;
		this.message = message;
		this.timestamp = System.currentTimeMillis();
	}
}
