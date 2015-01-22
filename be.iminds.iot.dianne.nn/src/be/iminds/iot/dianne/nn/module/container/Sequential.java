package be.iminds.iot.dianne.nn.module.container;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import be.iminds.iot.dianne.nn.module.Module;
import be.iminds.iot.dianne.tensor.Tensor;

public class Sequential implements Module {

	protected UUID id;
	protected List<Module> modules = new ArrayList<Module>();
	
	public Sequential(){
		this.id = UUID.randomUUID();
	}
	
	public Sequential(UUID id){
		this.id = id;
	}
	
	@Override
	public UUID getId() {
		return id;
	}

	@Override
	public void forward(UUID moduleId, Tensor input) {
		modules.get(0).forward(moduleId, input);
	}

	@Override
	public void backward(UUID moduleId, Tensor gradOutput) {
		modules.get(modules.size()-1).backward(moduleId, gradOutput);
	}

	@Override
	public void setNext(Module... next) {
		modules.get(modules.size()-1).setNext(next);
	}

	@Override
	public void setPrevious(Module... prev) {
		modules.get(0).setPrevious(prev);
	}

	public void add(Module m){
		modules.add(m);
		if(modules.size()>1){
			Module m1 = modules.get(modules.size()-2);
			Module m2 = modules.get(modules.size()-1);
			
			m1.setNext(m2);
			m2.setPrevious(m1);
		}
	}
}
