package be.iminds.iot.dianne.nn.learn.criterion;

import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class BCECriterion implements Criterion {

	protected Tensor invOut;
	protected Tensor invTar;
	
	protected Tensor logOut;
	protected Tensor logInvOut;
	
	protected Tensor error;
	protected Tensor grad;
	
	@Override
	public Tensor error(Tensor output, Tensor target) {
		invOut = TensorOps.mul(invOut, output, -1);
		TensorOps.add(invOut, invOut, 1);
		invTar = TensorOps.mul(invTar, target, -1);
		TensorOps.add(invTar, invTar, 1);
		
		logOut = TensorOps.log(logOut, output);
		logInvOut = TensorOps.log(logInvOut, invOut);
		
		error = TensorOps.cmul(error, target, logOut);
		TensorOps.addcmul(error, error, 1, invTar, logInvOut);
		
		return new Tensor(new float[]{-TensorOps.sum(error)}, 1);
	}

	@Override
	public Tensor grad(Tensor output, Tensor target) {
		invOut = TensorOps.mul(invOut, output, -1);
		TensorOps.add(invOut, invOut, 1);
		
		grad = TensorOps.sub(grad, output, target);
		
		TensorOps.cdiv(grad, grad, output);
		TensorOps.cdiv(grad, grad, invOut);
		
		return grad;
	}

}
