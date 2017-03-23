package be.iminds.iot.dianne.launcher;

import java.util.List;

import aQute.bnd.build.Project;
import aQute.launcher.plugin.ProjectLauncherImpl;

@SuppressWarnings("restriction")
public class DianneLauncher extends ProjectLauncherImpl  {

	private GPUDetector gpuDetector = new NvidiaGPUDetector();
	
	public DianneLauncher(Project project) throws Exception {
		super(project);
	}

	@Override
	public String getSystemCapabilities() {
		StringBuilder builder  = new StringBuilder();

		String caps = super.getSystemCapabilities();
		if(caps != null)
			builder.append(caps);
		
		List<GPU> gpus = gpuDetector.availableGPUs();
		
		for(GPU gpu : gpus){
			if(builder.length() > 0){
				builder.append(",");
			}
			
			builder.append("osgi.native.gpu").append(";")
			   .append("osgi.native.gpu.vendor").append("=").append(gpu.vendor).append(";")
			   .append("osgi.native.gpu.model").append("=").append(gpu.model).append(";")
			   .append("osgi.native.gpu.memory").append("=").append(gpu.memory);
			   
			   gpu.properties.entrySet().forEach(e -> builder.append(";").append(e.getKey()).append("=").append("\"").append(e.getValue()).append("\""));
		}
		
		return builder.toString();
	}
}
