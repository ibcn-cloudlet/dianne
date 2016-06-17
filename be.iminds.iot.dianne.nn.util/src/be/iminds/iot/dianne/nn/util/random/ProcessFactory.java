package be.iminds.iot.dianne.nn.util.random;

public class ProcessFactory {
	
	private final DistributionFactory interArrivalFactory;
	
	public ProcessFactory(DistributionFactory interArrivalFactory){
		this.interArrivalFactory = interArrivalFactory;
	}
	
	public Process nextProcess(long seed, long start){
		return new Process(start,interArrivalFactory.nextDistribution(seed));
	}
	
	public Process nextProcess(long start){
		return new Process(start,interArrivalFactory.nextDistribution());
	}
	
	public Process nextProcess(){
		return new Process(interArrivalFactory.nextDistribution());
	}
}
