package be.iminds.iot.dianne.launcher;

import java.util.List;

/**
 * Main class for testing purposes
 * 
 * @author tverbele
 *
 */
public class Main {
	
	public static void main(String[] args){
		NvidiaGPUDetector d = new NvidiaGPUDetector();
		
		List<GPU> gpus = d.availableGPUs();
		if(gpus.size() == 0){
			System.out.println("No NVIDIA GPU devices found");
			return;
		}
		
		System.out.println("Available GPUs:");
		gpus.forEach(gpu -> System.out.println("* "+gpu.toString()));
		
	}
	
}
