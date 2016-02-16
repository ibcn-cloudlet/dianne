package be.iminds.iot.dianne.api.coordinator;

import java.util.List;
import java.util.Map;
import java.util.UUID;

public class Job {

	public final UUID id;
	public final String nn;
	public final String dataset;
	public final Map<String, String> config;
	
	public Job(UUID id, String nn, String d, Map<String, String> c){
		this.id = id;
		this.nn = nn;
		this.dataset = d;
		this.config = c;
	}
	
	public long submitted;
	public long started;
	public long stopped;
	public List<UUID> targets;
	
}
