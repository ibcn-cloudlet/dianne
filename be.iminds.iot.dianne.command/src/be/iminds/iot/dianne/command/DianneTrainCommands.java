package be.iminds.iot.dianne.command;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.DatasetRangeAdapter;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.train.Evaluation;
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
	
	public void eval(String nnId, String dataset, int start, int end){
		Dataset d = commands.datasets.get(dataset);
		if(d==null){
			System.out.println("Dataset "+dataset+" not available");
			return;
		}

		Input input = commands.getInput(nnId);
		if(input==null){
			System.out.println("No Input module found for neural network "+nnId);
			return;
		}
		
		Output output = commands.getOutput(nnId);
		if(output==null){
			System.out.println("No Output module found for neural network "+nnId);
			return;
		}
		
		try {
			ArgMaxEvaluator evaluator = new ArgMaxEvaluator(commands.factory);
			DatasetRangeAdapter range = new DatasetRangeAdapter(d, start, end);
			
			Evaluation eval = evaluator.evaluate(input, output, range);
			System.out.println("Overall accuracy: "+eval.accuracy());
		
		} catch(Throwable t){
			t.printStackTrace();
		} 
	}
}
