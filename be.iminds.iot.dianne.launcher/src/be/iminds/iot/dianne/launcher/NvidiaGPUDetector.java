package be.iminds.iot.dianne.launcher;

import java.util.ArrayList;
import java.util.List;

public class NvidiaGPUDetector implements GPUDetector {

	@Override
	public List<GPU> availableGPUs() {

		// TODO fetch GPU info from system...
		
		GPU gpu = new GPU();
		gpu.vendor = "NVIDIA";
		gpu.model = "GTX 980";
		gpu.memory = 1024;
		
		gpu.properties.put("com.nvidia.cudaVersion:List<Version>", "2.0,2.1,2.2,2.3,3.0,3.1,3.2,4.0,4.1,4.2,5.0,5.5,6.0,6.5,7.0,7.5,8.0");
		gpu.properties.put("com.nvidia.computecapability:Version", "7.0");
		
		List<GPU> gpus = new ArrayList<>();
		gpus.add(gpu);
		return gpus;
	}
}
