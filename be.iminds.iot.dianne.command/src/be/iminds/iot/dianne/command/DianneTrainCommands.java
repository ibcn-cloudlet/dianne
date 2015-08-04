package be.iminds.iot.dianne.command;

import be.iminds.iot.dianne.api.dataset.DatasetRangeAdapter;
import be.iminds.iot.dianne.api.nn.train.api.Evaluation;
import be.iminds.iot.dianne.nn.train.eval.ArgMaxEvaluator;

// helper class to separate training related commands 
// that can be turned off when training bundle not deployed
// ugly, but works
public class DianneTrainCommands {
	
	final DianneCommands commands;
	
	public DianneTrainCommands(DianneCommands c) throws NoClassDefFoundError {
		this.commands = c;
		
		// to test whether this class is available, will throw exception otherwise
		ArgMaxEvaluator test = new ArgMaxEvaluator(c.factory);
	}
	
	public void eval(int start, int end){
		if(commands.network==null){
			System.out.println("No neural network loaded, load one first with loadNN");
			return;
		} 
		if(commands.dataset==null){
			System.out.println("No dataset loaded, load one first with loadDataset");
			return;
		}
		if(commands.input==null){
			System.out.println("Loaded neural network has no valid Input module");
			return;
		}
		
		ArgMaxEvaluator evaluator = new ArgMaxEvaluator(commands.factory);
		DatasetRangeAdapter range = new DatasetRangeAdapter(commands.dataset, start, end);
		Evaluation eval = evaluator.evaluate(commands.input, commands.output, range);
		System.out.println("Overall accuracy: "+eval.accuracy());
	}
}
