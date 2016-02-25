package be.iminds.iot.dianne.api.coordinator;

public class Status {

	public final int queued;
	public final int running;
	
	public final int learn;
	public final int eval;
	public final int act;
	public final int idle;

	public final long spaceLeft;
	
	public final long bootTime;
	
	public Status(int queued, int running,
			int learn, int eval, int act, int idle,
			long spaceLeft, long bootTime){
		this.queued = queued;
		this.running = running;
		this.learn = learn;
		this.eval = eval;
		this.act = act;
		this.idle = idle;
		this.spaceLeft = spaceLeft;
		this.bootTime = bootTime;
	}
	
}
