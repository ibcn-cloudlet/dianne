package be.iminds.iot.dianne.api.nn.eval;

import java.util.Map;

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

	Evaluation eval(NeuralNetworkInstanceDTO nni, String dataset, Map<String, String> config) throws Exception;

}
