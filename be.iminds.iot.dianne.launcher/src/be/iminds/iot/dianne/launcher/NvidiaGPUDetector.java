package be.iminds.iot.dianne.launcher;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.SortedMap;
import java.util.TreeMap;

import aQute.bnd.version.Version;


public class NvidiaGPUDetector implements GPUDetector {

	final SortedMap<Version, Version> cudaVersion_driverVersion = new TreeMap<>();
	
	public NvidiaGPUDetector() {
		cudaVersion_driverVersion.put(new Version("2.0"), new Version("0.0")); // TODO cuda 2.0 supported from which driver version on? 
		cudaVersion_driverVersion.put(new Version("3.0"), new Version("195.36.15"));
		cudaVersion_driverVersion.put(new Version("3.1"), new Version("256.40"));
		cudaVersion_driverVersion.put(new Version("3.2"), new Version("260.19.26"));
		cudaVersion_driverVersion.put(new Version("4.0"), new Version("270.41.19"));
		cudaVersion_driverVersion.put(new Version("4.1"), new Version("285.05.33"));
		cudaVersion_driverVersion.put(new Version("4.2"), new Version("295.41"));
		cudaVersion_driverVersion.put(new Version("5.0"), new Version("304"));
		cudaVersion_driverVersion.put(new Version("5.5"), new Version("319"));
		cudaVersion_driverVersion.put(new Version("6.0"), new Version("331"));
		cudaVersion_driverVersion.put(new Version("6.5"), new Version("340"));
		cudaVersion_driverVersion.put(new Version("7.0"), new Version("346"));
		cudaVersion_driverVersion.put(new Version("7.5"), new Version("352"));
		cudaVersion_driverVersion.put(new Version("8.0"), new Version("367.40"));
	}
	
	@Override
	public List<GPU> availableGPUs() {
		List<GPU> gpus = new ArrayList<>();

		// try to get gpu info from nvidia-smi command
		// TODO use deviceQuery to get additional info (e.g. compute capability?)
		try(BufferedReader in 
				= new BufferedReader(new InputStreamReader(
					Runtime.getRuntime().exec("nvidia-smi").getInputStream()))){
			// skip first two lines
			in.readLine(); // date
			in.readLine(); // header top ---
			String driver = in.readLine().substring(52, 60).trim();
			in.readLine(); // header bottom ---
			in.readLine(); // stats info
			in.readLine(); // stats info
			in.readLine(); // ===
			
			boolean more = true;
			do {
				String line1 = in.readLine();
				String line2 = in.readLine();
				in.readLine(); // +----+
			
				if(!line1.contains("|")){
					more = false;
					break;
				}
				
				// model info
				String model = line1.substring(6, 27).trim();
				
				// total memory
				String memory = line2.substring(33, 54).trim();
				memory = memory.substring(memory.indexOf("/")+1, memory.length()-3).trim();
				
				GPU gpu = new GPU();
				gpu.vendor = "NVIDIA";
				gpu.model = model;
				gpu.memory = Integer.parseInt(memory);
				gpu.properties.put("com.nvidia.cuda.driverVersion:Version", driver);
				
				Version driverVersion = new Version(driver);
				StringBuilder cudaVersions = new StringBuilder();
				cudaVersion_driverVersion.entrySet().forEach(e -> {
					if(driverVersion.compareTo(e.getValue()) > 0){
						cudaVersions.append(e.getKey()).append(",");
					}
				});
				cudaVersions.deleteCharAt(cudaVersions.length()-1);
				gpu.properties.put("com.nvidia.cuda.version:List<Version>", cudaVersions.toString());
				
				gpus.add(gpu);
			} while(more);
			
		} catch (IOException e) {
		}
		
		return gpus;
	}
}
