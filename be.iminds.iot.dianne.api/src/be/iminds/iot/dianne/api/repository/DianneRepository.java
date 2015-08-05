package be.iminds.iot.dianne.api.repository;

import java.io.IOException;
import java.util.List;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;

public interface DianneRepository {

	List<String> avialableNeuralNetworks();
	
	NeuralNetworkDTO loadNeuralNetwork(String nnName) throws IOException;
	
	void storeNeuralNetwork(NeuralNetworkDTO nn);
	
	String loadLayout(String nnName) throws IOException;
	
	void storeLayout(String nnName, String layout);

	float[] loadWeights(UUID moduleId) throws IOException;
	
	void storeWeights(UUID moduleId, float[] weights);
}
