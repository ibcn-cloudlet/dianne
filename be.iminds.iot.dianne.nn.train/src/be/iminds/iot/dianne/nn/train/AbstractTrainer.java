package be.iminds.iot.dianne.nn.train;

import java.util.ArrayList;
import java.util.List;

import be.iminds.iot.dianne.dataset.Dataset;
import be.iminds.iot.dianne.nn.module.Input;
import be.iminds.iot.dianne.nn.module.Module;
import be.iminds.iot.dianne.nn.module.Module.Mode;
import be.iminds.iot.dianne.nn.module.Output;
import be.iminds.iot.dianne.nn.module.Preprocessor;
import be.iminds.iot.dianne.nn.module.Trainable;

public abstract class AbstractTrainer implements Trainer  {

	
	public synchronized void train(final List<Module> modules, 
			final Criterion criterion, final Dataset data){
		Input input = null;
		Output output = null;
		List<Trainable> trainables = new ArrayList<Trainable>();
		List<Preprocessor> preprocessors = new ArrayList<Preprocessor>();
		
		for(Module m : modules){
			if(m instanceof Input){
				input = (Input) m;
			} else if(m instanceof Output){
				output = (Output) m;
			} else if(m instanceof Trainable){
				trainables.add((Trainable) m);
			} else if(m instanceof Preprocessor){
				preprocessors.add((Preprocessor) m);
			}
			
			m.setMode(Mode.TRAINING);
		}
		
		// TODO check if module list is valid for training
		assert input!=null;
		assert output!=null;
		
		// TODO maybe better separation between strategy and abstract trainer ?
		train(input, output, trainables, preprocessors, criterion, data);
		
		for(Module m : modules){
			m.setMode(Mode.EVALUATION);
		}
	}
	
	protected abstract void train(final Input input, final Output output, 
			final List<Trainable> module, final List<Preprocessor> preprocessors, 
			final Criterion criterion, final Dataset data);
}
