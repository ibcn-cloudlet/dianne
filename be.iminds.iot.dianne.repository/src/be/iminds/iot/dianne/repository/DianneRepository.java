package be.iminds.iot.dianne.repository;

import java.io.IOException;
import java.util.UUID;

public interface DianneRepository {

	public float[] loadWeights(UUID id) throws IOException;
	
	public void storeWeights(UUID id, float[] weights);
}
