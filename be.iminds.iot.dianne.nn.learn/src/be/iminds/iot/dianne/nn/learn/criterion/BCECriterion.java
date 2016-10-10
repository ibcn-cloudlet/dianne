package be.iminds.iot.dianne.nn.learn.criterion;

import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class BCECriterion implements Criterion {
	
	protected static final float EPS = 1e-12f;
	
	protected Tensor output;
	protected Tensor target;

	protected Tensor invOut;
	protected Tensor invTar;
	
	protected Tensor epsOut;
	protected Tensor epsInvOut;
	
	protected Tensor logOut;
	protected Tensor logInvOut;
	
	protected Tensor error;
	protected Tensor grad;
	
	@Override
	public Tensor loss(Tensor output, Tensor target) {
		this.output = output;
		this.target = target;
		
		invOut = TensorOps.mul(invOut, output, -1);
		TensorOps.add(invOut, invOut, 1);
		invTar = TensorOps.mul(invTar, target, -1);
		TensorOps.add(invTar, invTar, 1);
		
		epsOut = TensorOps.add(epsOut, output, EPS);
		epsInvOut = TensorOps.add(epsInvOut, invOut, EPS);
		
		logOut = TensorOps.log(logOut, epsOut);
		logInvOut = TensorOps.log(logInvOut, epsInvOut);
		
		error = TensorOps.cmul(error, target, logOut);
		TensorOps.addcmul(error, error, 1, invTar, logInvOut);
		
		return new Tensor(new float[]{-TensorOps.sum(error)}, 1);
	}

	@Override
	public Tensor grad(Tensor output, Tensor target) {
		if(!output.equals(this.output) || !target.equals(this.target)) {
			invOut = TensorOps.mul(invOut, output, -1);
			TensorOps.add(invOut, invOut, 1);
			
			epsOut = TensorOps.add(epsOut, output, EPS);
			epsInvOut = TensorOps.add(epsInvOut, invOut, EPS);
		}
		
		grad = TensorOps.sub(grad, output, target);
		
		TensorOps.cdiv(grad, grad, epsOut);
		TensorOps.cdiv(grad, grad, epsInvOut);
		
		return grad;
	}

}
