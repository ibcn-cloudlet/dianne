package be.iminds.iot.dianne.command;

import java.util.UUID;

import org.osgi.framework.ServiceReference;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.DatasetRangeAdapter;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Output;
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
	
	public void eval(String dataset, String nnId, int start, int end){
		Dataset d = commands.datasets.get(dataset);
		if(d==null){
			System.out.println("Dataset "+dataset+" not available");
			return;
		}

		UUID inputId = commands.getInputId(nnId);
		if(inputId==null){
			System.out.println("No Input module found for neural network "+nnId);
			return;
		}
		
		UUID outputId = commands.getOutputId(nnId);
		if(outputId==null){
			System.out.println("No Output module found for neural network "+nnId);
			return;
		}
		
		ServiceReference refInput = commands.getModule(UUID.fromString(nnId), inputId);
		if(refInput==null){
			System.out.println("Input module "+inputId+" not found");
			return;
		}
		
		ServiceReference refOutput = commands.getModule(UUID.fromString(nnId), outputId);
		if(refOutput==null){
			System.out.println("Output module "+outputId+" not found");
			return;
		}
		
		try {
			Input input = (Input) commands.context.getService(refInput);
			Output output = (Output) commands.context.getService(refOutput);
		
			ArgMaxEvaluator evaluator = new ArgMaxEvaluator(commands.factory);
			DatasetRangeAdapter range = new DatasetRangeAdapter(d, start, end);
			
			Evaluation eval = evaluator.evaluate(input, output, range);
			System.out.println("Overall accuracy: "+eval.accuracy());
		
		} catch(Throwable t){
			t.printStackTrace();
		} finally {
			commands.context.ungetService(refInput);
			commands.context.ungetService(refOutput);
		}
	}
}
