/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  iMinds - IBCN - UGent
 *
 * This file is part of DIANNE.
 *
 * DIANNE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.api.nn.eval;

import java.util.List;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

/**
 * Result of the evaluation of a Dataset, provides access to the confusion matrix.
 * 
 * @author tverbele
 *
 */
public class Evaluation {

	protected final TensorFactory factory;
	
	// the actual outputs 
	protected List<Tensor> outputs;
	// resulting confusion matrix
	protected Tensor confusionMatrix;
	// time to run the evaluation
	protected long time;
	
	public Evaluation(TensorFactory factory, Tensor confusionMatrix, List<Tensor> outputs, long time){
		//assert confusionMatrix.dim() == 2;
		//assert confusionMatrix.dims()[0] == confusionMatrix.dims()[1];
		this.confusionMatrix = confusionMatrix;
		this.outputs = outputs;
		this.factory = factory;
		this.time = time;
	}
	
	@Override
	public String toString(){
		return confusionMatrix.toString();
	}
	
	/**
	 * @return the total evaluation time
	 */
	public long evaluationTime(){
		return time;
	}
	
	/**
	 * @return average time for processing one sample
	 */
	public float forwardTime(){
		float total = factory.getTensorMath().sum(confusionMatrix);
		float sampleTime = time/total;
		return sampleTime;
	}
	
	public List<Tensor> getOutputs(){
		return outputs;
	}
	
	public Tensor getOutput(int index){
		return outputs.get(index);
	}
	
	/**
	 * @return the confusion matrix
	 */
	public Tensor getConfusionMatix(){
		return confusionMatrix;
	}
	
	/**
	 * @return accuracy on global dataset
	 */
	public float accuracy(){
		float tp = factory.getTensorMath().sum(confusionMatrix.diag(null));
		float total = factory.getTensorMath().sum(confusionMatrix);
		return tp/total;
	}
	
	/**
	 * @return error on global dataset
	 */
	public float error() {
		return 1-accuracy();
	}

	/**
	 * @return true positives of i-th class
	 */
	public float tp(int i) {
		return confusionMatrix.get(i,i);
	}

	/**
	 * @return false positives of i-th class
	 */
	public float fp(int i) {
		return factory.getTensorMath().sum(confusionMatrix.select(1, i))-confusionMatrix.get(i,i);
	}

	/**
	 * @return false negatives of i-th class
	 */
	public float fn(int i) {
		return factory.getTensorMath().sum(confusionMatrix.select(0, i))-confusionMatrix.get(i,i);
	}

	/**
	 * @return true negatives of i-th class
	 */
	public float tn(int i) {
		return factory.getTensorMath().sum(confusionMatrix)-fn(i)-fp(i)-tp(i);
	}

	/**
	 * @return sensitivity of i-th class (hit rate)
	 */
	public float sensitivity(int i) {
		return tp(i)/(tp(i)+fn(i));
	}

	/**
	 * @return specificity of i-th class (true negative rate)
	 */
	public float specificity(int i) {
		return tn(i)/(tn(i)+fp(i));
	}

	/**
	 * @return precision of i-th class (positive predicted value)
	 */
	public float precision(int i) {
		return tp(i)/(tp(i)+fp(i));
	}

	/**
	 * @return negative predictive value of i-th class
	 */
	public float npv(int i) {
		return tn(i)/(tn(i)+fn(i));
	}

	/**
	 * @return fall-out of i-th class (false positive rate)
	 */
	public float fallout(int i) {
		return fp(i)/(fp(i)+tn(i));
	}

	/**
	 * @return false discovery rate of i-th class
	 */
	public float fdr(int i) {
		return fp(i)/(fp(i)+tp(i));
	}

	/**
	 * @return false negative rate of i-th class (miss rate)
	 */
	public float fnr(int i) {
		return fn(i)/(fn(i)+tp(i));
	}

	/**
	 * @return accuracy of i-th class
	 */
	public float accuracy(int i) {
		return (tp(i)+tn(i))/factory.getTensorMath().sum(confusionMatrix);
	}

	/**
	 * @return error of i-th class
	 */
	public float error(int i) {
		return (fp(i)+fn(i))/factory.getTensorMath().sum(confusionMatrix);
	}

	/**
	 * @return f1 score of i-th class
	 */
	public float f1(int i) {
		return 2*tp(i)/(2*tp(i)+fp(i)+fn(i));
	}

	/**
	 * @return matthews correlation coefficient of i-th class
	 */
	public float mcc(int i) {
		return (tp(i)*tn(i) - fp(i)*fn(i))/(float)Math.sqrt((tp(i)+fp(i))*(tp(i)+fn(i))*(tn(i)+fp(i))*(tn(i)+fn(i)));
	}


}
