package be.iminds.iot.dianne.api.repository;

import java.io.IOException;
import java.util.List;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;

public interface DianneRepository {

	public List<String> networks();
	
	public NeuralNetworkDTO loadNetwork(String network) throws IOException;
	
	public void storeNetwork(String network, String modules);
	
	public String loadLayout(String network) throws IOException;
	
	public void storeLayout(String network, String layout);

	public float[] loadWeights(UUID id) throws IOException;
	
	public void storeWeights(UUID id, float[] weights);
}
