package be.iminds.iot.dianne.api.repository;

import java.io.IOException;
import java.util.List;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;

public interface DianneRepository {

	List<String> avialableNeuralNetworks();
	
	NeuralNetworkDTO loadNeuralNetwork(String nnName) throws IOException;
	
	void storeNeuralNetwork(NeuralNetworkDTO nn);
	

	float[] loadParameters(UUID moduleId) throws IOException;
	
	void storeParameters(UUID moduleId, float[] weights);
	
	
	// these are some helper methods for saving the jsplumb layout of the UI builder
	// of utterly no importance for the rest and can be ignored...
	String loadLayout(String nnName) throws IOException;
	
	void storeLayout(String nnName, String layout);
}
