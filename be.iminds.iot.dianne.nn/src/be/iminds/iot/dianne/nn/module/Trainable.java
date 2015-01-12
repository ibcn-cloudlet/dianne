package be.iminds.iot.dianne.nn.module;

public interface Trainable {

	public void accGradParameters();
	
	public void zeroGradParameters();
	
	public void updateParameters(final float learningRate);
	
}
