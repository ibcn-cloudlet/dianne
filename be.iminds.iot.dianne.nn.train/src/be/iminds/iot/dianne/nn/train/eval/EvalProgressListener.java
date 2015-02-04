package be.iminds.iot.dianne.nn.train.eval;

import be.iminds.iot.dianne.tensor.Tensor;

public interface EvalProgressListener {

	public void onProgress(Tensor confusionMatrix);
}
