package be.iminds.iot.dianne.nn.train.strategy;

public interface TrainProgressListener {

	public void onProgress(int epoch, int sample, float error);
	
}
