package be.iminds.iot.dianne.api.nn.eval;

import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;

/**
 * The Evaluator evaluates a neural network instance on a (portion of a) dataset.
 * 
 *  It collects metrics such as the accuracy and the forward time.
 * 
 * @author tverbele
 *
 */
public interface Evaluator {

	/**
	 * @return uuid of this evaluator - same as the frameworkId this evaluator is deployed on
	 */
	UUID getEvaluatorId();
	
	/**
	 * Evaluate a neural network instance on a (portion of a) dataset
	 * @param nni neural network instance to evaluate
	 * @param dataset dataset to evaluate the nni on
	 * @param config configuration
	 * @return Evaluation 
	 * @throws Exception
	 */
	Evaluation eval(NeuralNetworkInstanceDTO nni, String dataset, Map<String, String> config) throws Exception;

}
