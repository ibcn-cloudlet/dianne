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
		id = UUID.randomUUID(); // TODO meaningful id?
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
	public void addNext(Module... next) {
		modules.get(modules.size()-1).addNext(next);
	}

	@Override
	public void removeNext(Module... next) {
		modules.remove(modules.size()-1).addNext(next);
	}

	@Override
	public void addPrevious(Module... prev) {
		modules.get(0).addPrevious(prev);
	}

	@Override
	public void removePrevious(Module... prev) {
		modules.get(0).removePrevious(prev);
	}

	public void add(Module m){
		modules.add(m);
		if(modules.size()>1){
			Module m1 = modules.get(modules.size()-2);
			Module m2 = modules.get(modules.size()-1);
			
			m1.addNext(m2);
			m2.addPrevious(m1);
		}
	}
}
