package be.iminds.iot.dianne.api.nn.eval;

import java.util.UUID;

public interface EvaluatorListener {

	void onProgress(UUID evaluatorId, EvaluationProgress p);
	
	void onException(UUID evaluatorId, Throwable e);
	
	void onFinish(UUID evaluatorId, EvaluationProgress p);

}
