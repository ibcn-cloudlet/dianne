package be.iminds.iot.dianne.nn.util.random;

import java.util.Random;

public abstract class Distribution {
	
	protected final Random random;
	
	public Distribution(long seed){
		random = new HighQualityRandom(seed);
	}
	
	public Distribution(){
		random = new HighQualityRandom();
	}
	
	public abstract double nextDouble();
	
	public long nextLong(){
		return (long) this.nextDouble();
	}
	
	public int nextInt(){
		return (int) this.nextDouble();
	}
}
