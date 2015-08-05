package be.iminds.iot.dianne.api.nn.platform;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;

public interface NeuralNetworkManager {

	NeuralNetworkInstanceDTO deployNeuralNetwork(String name, UUID runtimeId) throws InstantiationException;
	
	NeuralNetworkInstanceDTO deployNeuralNetwork(String name, UUID runtimeId, Map<UUID, UUID> deployment) throws InstantiationException;

	void undeployNeuralNetwork(NeuralNetworkInstanceDTO nn);
	
	List<NeuralNetworkInstanceDTO> getNeuralNetworks();
	
	List<String> getSupportedNeuralNetworks();
	
	List<UUID> getRuntimes();

}
