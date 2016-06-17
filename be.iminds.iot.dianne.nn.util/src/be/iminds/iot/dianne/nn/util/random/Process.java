package be.iminds.iot.dianne.nn.util.random;

public class Process {
	
	private final Distribution interArrival;
	
	private long start;
	private long current;
	
	public Process(Distribution interArrival){
		this(0l,interArrival);
	}
	
	public Process(long start, Distribution interArrival){
		this.start = start;
		this.current = this.start;
		this.interArrival = interArrival;
	}
	
	public Process(Process process){
		this(process.start,process.interArrival);
	}
	
	public long nextEvent(){
		return this.current += interArrival.nextLong();
	}
	
	public void reset(){
		reset(start);
	}
	
	public void reset(long start){
		this.start = start;
		this.current = this.start;
	}
}
