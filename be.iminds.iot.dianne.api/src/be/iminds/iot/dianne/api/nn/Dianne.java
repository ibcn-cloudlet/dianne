package be.iminds.iot.dianne.api.nn;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;

/**
 * Convenient service to create NeuralNetwork services from instance dto
 * 
 * @author tverbele
 *
 */
public interface Dianne {

	NeuralNetwork getNeuralNetwork(NeuralNetworkInstanceDTO nn);
	
}