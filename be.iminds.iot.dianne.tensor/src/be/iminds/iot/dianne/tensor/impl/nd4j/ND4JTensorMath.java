package be.iminds.iot.dianne.tensor.impl.nd4j;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarGreaterThan;
import org.nd4j.linalg.api.ops.impl.transforms.Exp;
import org.nd4j.linalg.api.ops.impl.transforms.Log;
import org.nd4j.linalg.api.ops.impl.transforms.OneMinus;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.SigmoidDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.Tanh;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.GreaterThan;
import org.nd4j.linalg.convolution.Convolution.Type;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import be.iminds.iot.dianne.tensor.TensorMath;

public class ND4JTensorMath implements TensorMath<ND4JTensor> {

	@Override
	public ND4JTensor add(ND4JTensor res, ND4JTensor tensor, float value) {
		if(res==null){
			return new ND4JTensor(tensor.data.add(value));
		} else {
			tensor.data.addi(value, res.data);
			return res;
		}
	}

	@Override
	public ND4JTensor add(ND4JTensor res, ND4JTensor tensor1, ND4JTensor tensor2) {
		if(res==null){
			return new ND4JTensor(tensor1.data.add(tensor2.data));
		} else {
			tensor1.data.addi(tensor2.data, res.data);
			return res;
		}
	}

	@Override
	public ND4JTensor add(ND4JTensor res, ND4JTensor tensor1, float value,
			ND4JTensor tensor2) {
		if(res==null){
			INDArray r = tensor2.data.mul(value);
			r.addi(tensor1.data);
			return new ND4JTensor(r.reshape(tensor1.dims()));
		} else {
			tensor2.data.muli(value, res.data);
			res.data.addi(tensor1.data);
			res.reshape(tensor1.dims());
			return res;
		}
	}

	@Override
	public ND4JTensor sub(ND4JTensor res, ND4JTensor tensor, float value) {
		if(res==null){
			return new ND4JTensor(tensor.data.sub(value));
		} else {
			tensor.data.subi(value, res.data);
			return res;
		}
	}

	@Override
	public ND4JTensor sub(ND4JTensor res, ND4JTensor tensor1, ND4JTensor tensor2) {
		if(res==null){
			return new ND4JTensor(tensor1.data.sub(tensor2.data));
		} else {
			tensor1.data.subi(tensor2.data, res.data);
			return res;
		}
	}

	@Override
	public ND4JTensor sub(ND4JTensor res, ND4JTensor tensor1, float value,
			ND4JTensor tensor2) {
		if(res==null){
			INDArray r = tensor2.data.mul(value);
			tensor1.data.subi(r, r);
			return new ND4JTensor(r.reshape(tensor1.dims()));
		} else {
			tensor2.data.muli(value, res.data);
			tensor1.data.subi(res.data, res.data);
			res.reshape(tensor1.dims());
			return res;
		}
	}

	@Override
	public ND4JTensor mul(ND4JTensor res, ND4JTensor tensor1, float value) {
		if(res==null){
			return new ND4JTensor(tensor1.data.mul(value));
		} else {
			tensor1.data.mul(value, res.data);
			return res;
		}
	}

	@Override
	public ND4JTensor cmul(ND4JTensor res, ND4JTensor tensor1,
			ND4JTensor tensor2) {
		if(res==null){
			return new ND4JTensor(tensor1.data.mul(tensor2.data));
		} else {
			tensor1.data.mul(tensor2.data, res.data);
			return res;
		}
	}

	@Override
	public ND4JTensor div(ND4JTensor res, ND4JTensor tensor1, float value) {
		if(res==null){
			return new ND4JTensor(tensor1.data.div(value));
		} else {
			tensor1.data.div(value, res.data);
			return res;
		}
	}

	@Override
	public ND4JTensor cdiv(ND4JTensor res, ND4JTensor tensor1,
			ND4JTensor tensor2) {
		if(res==null){
			return new ND4JTensor(tensor1.data.div(tensor2.data));
		} else {
			tensor1.data.div(tensor2.data, res.data);
			return res;
		}
	}

	@Override
	public float dot(ND4JTensor vec1, ND4JTensor vec2) {
		return vec1.data.linearView().mmul(vec2.data.linearView().transpose()).getFloat(0);
	}

	@Override
	public ND4JTensor vv(ND4JTensor res, ND4JTensor vec1, ND4JTensor vec2) {
		if(res==null){
			return new ND4JTensor(vec1.data.linearView().mmul(vec2.data.linearView()));
		} else {
			vec1.data.linearView().transposei().mmuli(vec2.data.linearView(), res.data);
			return res;
		}
	}

	@Override
	public ND4JTensor mv(ND4JTensor res, ND4JTensor mat, ND4JTensor vec) {
		if(res==null){
			return new ND4JTensor(mat.data.mmul(vec.data.linearView().transpose()));
		} else {
			mat.data.mmuli(vec.data.linearView().transposei(), res.data);
			return res;
		}
	}

	@Override
	public ND4JTensor tmv(ND4JTensor res, ND4JTensor mat, ND4JTensor vec) {
		// TODO extra transpose here due to bug in ND4J
		if(res==null){
			return new ND4JTensor(mat.data.transpose().mmul(vec.data.linearView().transpose()).transpose());
		} else {
			mat.data.transpose().mmuli(vec.data.linearView().transpose(), res.data);
			res.data.transposei();
			return res;
		}
	}

	@Override
	public ND4JTensor mm(ND4JTensor res, ND4JTensor mat1, ND4JTensor mat2) {
		// TODO extra transpose here due to bug in ND4J
		if(res==null){
			return new ND4JTensor(mat1.data.mmul(mat2.data).transpose());
		} else {
			mat1.data.mmuli(mat2.data, res.data);
			res.data.transposei();
			return res;
		}
	}

	@Override
	public ND4JTensor addvv(ND4JTensor res, ND4JTensor mat, ND4JTensor vec1,
			ND4JTensor vec2) {
		ND4JTensor t = vv(res, vec1, vec2);
		return add(res, mat, t);
	}

	@Override
	public ND4JTensor addmv(ND4JTensor res, ND4JTensor vec1, ND4JTensor mat,
			ND4JTensor vec2) {
		ND4JTensor t = mv(res, mat, vec2);
		return add(res, vec1, t);
	}

	@Override
	public ND4JTensor addmm(ND4JTensor res, ND4JTensor mat, ND4JTensor mat1,
			ND4JTensor mat2) {
		ND4JTensor t = mm(res, mat1, mat2);
		return add(res, mat, t);
	}

	@Override
	public ND4JTensor exp(ND4JTensor res, ND4JTensor tensor) {
		if(res==null){
			TransformOp t = new Exp(tensor.data, tensor.data.dup());
			return new ND4JTensor(Nd4j.getExecutioner().execAndReturn(t));
		} else {
			TransformOp t = new Exp(tensor.data, res.data);
			Nd4j.getExecutioner().exec(t);
			return res;
		}
	}

	@Override
	public ND4JTensor log(ND4JTensor res, ND4JTensor tensor) {
		if(res==null){
			TransformOp t = new Log(tensor.data, tensor.data.dup());
			return new ND4JTensor(Nd4j.getExecutioner().execAndReturn(t));
		} else {
			TransformOp t = new Log(tensor.data, res.data);
			Nd4j.getExecutioner().exec(t);
			return res;
		}
	}

	@Override
	public ND4JTensor tanh(ND4JTensor res, ND4JTensor tensor) {
		if(res==null){
			TransformOp t = new Tanh(tensor.data, tensor.data.dup());
			return new ND4JTensor(Nd4j.getExecutioner().execAndReturn(t));
		} else {
			TransformOp t = new Tanh(tensor.data, res.data);
			Nd4j.getExecutioner().exec(t);
			return res;
		}
	}

	@Override
	public ND4JTensor dtanh(ND4JTensor res, ND4JTensor tensor) {
		if(res==null){
			TransformOp t = new OneMinus(tensor.data, tensor.data.dup());
			return new ND4JTensor(Nd4j.getExecutioner().execAndReturn(t));
		} else {
			TransformOp t = new OneMinus(tensor.data, res.data);
			Nd4j.getExecutioner().exec(t);
			return res;
		}
	}

	@Override
	public ND4JTensor sigmoid(ND4JTensor res, ND4JTensor tensor) {
		if(res==null){
			TransformOp t = new Sigmoid(tensor.data, tensor.data.dup());
			return new ND4JTensor(Nd4j.getExecutioner().execAndReturn(t));
		} else {
			TransformOp t = new Sigmoid(tensor.data, res.data);
			Nd4j.getExecutioner().exec(t);
			return res;
		}
	}

	@Override
	public ND4JTensor dsigmoid(ND4JTensor res, ND4JTensor tensor) {
		if(res==null){
			TransformOp t = new SigmoidDerivative(tensor.data, tensor.data.dup());
			return new ND4JTensor(Nd4j.getExecutioner().execAndReturn(t));
		} else {
			TransformOp t = new SigmoidDerivative(tensor.data, res.data);
			Nd4j.getExecutioner().exec(t);
			return res;
		}
	}

	@Override
	public ND4JTensor thresh(ND4JTensor res, ND4JTensor tensor, float thresh,
			float coeff, float offset) {
		// TODO make custom operation?
		ND4JTensor ret;
		if(res == null){
			INDArray d = tensor.data.mul(coeff);
			d.addi(offset);
			ret = new ND4JTensor(d);
		} else {
			mul(res, tensor, coeff);
			ret = add(res, res, offset);
		}
		
		ret.data.muli(Nd4j.getExecutioner().execAndReturn(new ScalarGreaterThan(tensor.data, thresh)));
		return ret;
	}

	@Override
	public ND4JTensor thresh(ND4JTensor res, ND4JTensor tensor,
			ND4JTensor threshs, ND4JTensor coeffs, ND4JTensor offsets) {
		// TODO make custom operation?
		ND4JTensor ret;
		if(res == null){
			INDArray d = tensor.data.mul(coeffs.data);
			d.addi(offsets.data);
			ret = new ND4JTensor(d);
		} else {
			cmul(res, tensor, coeffs);
			ret = add(res, res, offsets);
		}
		
		ret.data.muli(Nd4j.getExecutioner().execAndReturn(new GreaterThan(tensor.data, threshs.data, tensor.data, tensor.data.length())));
		return ret;
	}

	@Override
	public ND4JTensor dthresh(ND4JTensor res, ND4JTensor tensor, float thresh,
			float coeff) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor dthresh(ND4JTensor res, ND4JTensor tensor,
			ND4JTensor threshs, ND4JTensor coeffs) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor softmax(ND4JTensor res, ND4JTensor tensor) {
		if(res==null){
			TransformOp t = new SoftMax(tensor.data, tensor.data.dup());
			return new ND4JTensor(Nd4j.getExecutioner().execAndReturn(t));
		} else {
			TransformOp t = new SoftMax(tensor.data, res.data);
			Nd4j.getExecutioner().exec(t);
			return res;
		}
	}

	@Override
	public float sum(ND4JTensor tensor) {
		return tensor.data.linearView().sum(1).getFloat(0);
	}

	@Override
	public float max(ND4JTensor tensor) {
		return tensor.data.linearView().max(1).getFloat(0);
	}

	@Override
	public float min(ND4JTensor tensor) {
		return tensor.data.linearView().min(1).getFloat(0);
	}

	@Override
	public float mean(ND4JTensor tensor) {
		return tensor.data.linearView().mean(1).getFloat(0);
	}

	@Override
	public int argmax(ND4JTensor tensor) {
		float max = -Float.MAX_VALUE;
		int index = -1;
		INDArray data = tensor.data.linearView();
		for(int i=0;i<data.length();i++){
			float val =  data.getFloat(i);
			if(val > max){
				max = val;
				index = i;
			}
		}
		return index;
	}

	@Override
	public int argmin(ND4JTensor tensor) {
		float min = Float.MAX_VALUE;
		int index = -1;
		INDArray data = tensor.data.linearView();
		for(int i=0;i<data.length();i++){
			float val =  data.getFloat(i);
			if(val < min){
				min = val;
				index = i;
			}
		}
		return index;
	}

	@Override
	public ND4JTensor convolution2D(ND4JTensor res, ND4JTensor mat1,
			ND4JTensor mat2, int sx, int sy, int mode, boolean flip) {
		// TODO this should be done better?
		INDArray r;
		if(mode==1){
			r = Nd4j.getConvolution().conv2d(mat1.data, mat2.data, Type.FULL);
		} else if(mode==2){
			r = Nd4j.getConvolution().conv2d(mat1.data, mat2.data, Type.SAME);
		} else {
			r = Nd4j.getConvolution().conv2d(mat1.data, mat2.data, Type.VALID);
		}
		
		if(sx>1 || sy >1){
			r = Transforms.downSample(r, new int[]{sx, sy});
		}
		
		if(res==null){
			return new ND4JTensor(r);
		} else {
			// TODO what here?
			res.data = r;
			return res;
		}
	}

	@Override
	public ND4JTensor maxpool2D(ND4JTensor res, ND4JTensor mat, int w, int h,
			int sx, int sy) {
		INDArray r = Transforms.maxPool(mat.data, new int[]{w, h}, false);
		
		if(sx>1 || sy >1){
			r = Transforms.downSample(r, new int[]{sx, sy});
		}
		
		if(res==null){
			return new ND4JTensor(r);
		} else {
			// TODO what here?
			res.data = r;
			return res;
		}
	}

	@Override
	public ND4JTensor dmaxpool2D(ND4JTensor res, ND4JTensor mat2,
			ND4JTensor mat1, int w, int h, int sx, int sy) {
		// TODO Auto-generated method stub
		return null;
	}

}
