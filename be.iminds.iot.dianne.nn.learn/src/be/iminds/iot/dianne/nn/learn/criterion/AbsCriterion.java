package be.iminds.iot.dianne.nn.learn.criterion;

import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * Absolute difference error criterion
 * 
 * @author smbohez
 *
 */
public class AbsCriterion implements Criterion {
	
	protected Tensor diff;
	protected Tensor absdiff;
	protected Tensor error;
	protected Tensor grad;
	
	public AbsCriterion() {
		this.error = new Tensor(1);
	}
	
	@Override
	public Tensor error(final Tensor output, final Tensor target) {
		diff = TensorOps.sub(diff, output, target);
		absdiff = TensorOps.abs(absdiff, diff);
		error.set(TensorOps.sum(absdiff) / (output.dim() == 2 ? output.size(1) : output.size(0)), 0);
		return error;
	}

	@Override
	public Tensor grad(final Tensor output, final Tensor target) {
		grad = TensorOps.sign(grad, diff);
		grad = TensorOps.mul(grad, grad, 1.0f / (output.dim() == 2 ? output.size(1) : output.size(0)));
		return grad;
	}
}
