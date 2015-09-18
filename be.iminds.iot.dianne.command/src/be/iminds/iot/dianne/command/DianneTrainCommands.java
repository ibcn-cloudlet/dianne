package be.iminds.iot.dianne.command;

import java.util.UUID;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.DatasetRangeAdapter;
import be.iminds.iot.dianne.api.nn.platform.NeuralNetwork;
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

		NeuralNetwork nn = commands.dianne.getNeuralNetwork(UUID.fromString(nnId));
		if(nn==null){
			System.out.println("Neural network instance "+nnId+" not available");
			return;
		}
		
		try {
			ArgMaxEvaluator evaluator = new ArgMaxEvaluator(commands.factory);
			DatasetRangeAdapter range = new DatasetRangeAdapter(d, start, end);
			
			Evaluation eval = evaluator.evaluate(nn.getInput(), nn.getOutput(), range);
			System.out.println("Overall accuracy: "+eval.accuracy());
		
		} catch(Throwable t){
			t.printStackTrace();
		} 
	}
}
