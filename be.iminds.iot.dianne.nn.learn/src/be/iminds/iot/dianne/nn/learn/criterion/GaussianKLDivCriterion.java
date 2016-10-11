package be.iminds.iot.dianne.nn.learn.criterion;

import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory.BatchConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class GaussianKLDivCriterion implements Criterion {
	
	protected Tensor meanDiff;
	protected Tensor stdevRatio;
	protected Tensor logStdevRatio;
	
	protected Tensor sqTarStdev;
	protected Tensor invOutStdev;
	
	protected Tensor loss;
	protected Tensor grad;

	protected BatchConfig b;
	
	public GaussianKLDivCriterion(BatchConfig b) {
		this.b = b;
	}
	
	@Override
	public float loss(Tensor output, Tensor target) {
		int dim = output.dim()-1;
		int size = output.size(dim)/2;
		
		Tensor outMean = output.narrow(dim, 0, size);
		Tensor outStdev = output.narrow(dim, size, size);
		Tensor tarMean = target.narrow(dim, 0, size);
		Tensor tarStdev = target.narrow(dim, size, size);
		
		meanDiff = TensorOps.sub(meanDiff, tarMean, outMean);
		loss = TensorOps.cdiv(loss, meanDiff, tarStdev);
		TensorOps.cmul(loss, loss, loss);
		
		stdevRatio = TensorOps.cdiv(stdevRatio, outStdev, tarStdev);
		TensorOps.cmul(stdevRatio, stdevRatio, stdevRatio);
		TensorOps.add(loss, loss, stdevRatio);
		
		logStdevRatio = TensorOps.log(logStdevRatio, stdevRatio);
		TensorOps.sub(loss, loss, logStdevRatio);
		
		TensorOps.sub(loss, loss, 1);
		TensorOps.div(loss, loss, 2);
		
		if(b.batchAverage){
			return TensorOps.sum(loss)/b.batchSize;
		} else {
			return TensorOps.sum(loss);
		}
	}

	@Override
	public Tensor grad(Tensor output, Tensor target) {
		int dim = output.dim()-1;
		int size = output.size(dim)/2;
		
		grad = output.copyInto(grad);
		
		Tensor outStdev = output.narrow(dim, size, size);
		Tensor tarMean = target.narrow(dim, 0, size);
		Tensor tarStdev = target.narrow(dim, size, size);
		Tensor gradMean = grad.narrow(dim, 0, size);
		Tensor gradStdev = grad.narrow(dim, size, size);
		
		sqTarStdev = TensorOps.cmul(sqTarStdev, tarStdev, tarStdev);
		invOutStdev = TensorOps.pow(invOutStdev, outStdev, -1);
		
		TensorOps.sub(gradMean, gradMean, tarMean);
		TensorOps.cdiv(gradMean, gradMean, sqTarStdev);
		
		TensorOps.cdiv(gradStdev, gradStdev, sqTarStdev);
		TensorOps.sub(gradStdev, gradStdev, invOutStdev);
		
		if(b.batchAverage){
			TensorOps.div(grad, grad, b.batchSize);
		}
			
		return grad;
	}

}
