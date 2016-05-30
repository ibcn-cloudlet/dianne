package be.iminds.iot.dianne.api.nn.learn;

import java.util.UUID;

public interface LearnerListener {

	void onProgress(UUID learnerId, LearnProgress p);
	
	void onException(UUID learnerId, Throwable e);
	
	void onFinish(UUID learnerId, LearnProgress p);
	
}
