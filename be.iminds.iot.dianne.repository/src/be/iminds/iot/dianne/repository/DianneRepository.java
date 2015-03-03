package be.iminds.iot.dianne.repository;

import java.io.IOException;
import java.util.List;
import java.util.UUID;

public interface DianneRepository {

	public List<String> networks();
	
	
	
	public float[] loadWeights(UUID id) throws IOException;
	
	public void storeWeights(UUID id, float[] weights);
}
