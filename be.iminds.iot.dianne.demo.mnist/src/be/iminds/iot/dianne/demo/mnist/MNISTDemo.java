package be.iminds.iot.dianne.demo.mnist;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.dataset.Dataset;
import be.iminds.iot.dianne.dataset.mnist.MNISTDataset;
import be.iminds.iot.dianne.dataset.mnist.MNISTDataset.Set;
import be.iminds.iot.dianne.nn.module.Input;
import be.iminds.iot.dianne.nn.module.Output;
import be.iminds.iot.dianne.nn.module.OutputListener;
import be.iminds.iot.dianne.nn.module.Trainable;
import be.iminds.iot.dianne.nn.train.Criterion;
import be.iminds.iot.dianne.nn.train.Evaluation;
import be.iminds.iot.dianne.nn.train.Evaluator;
import be.iminds.iot.dianne.nn.train.Trainer;
import be.iminds.iot.dianne.nn.train.criterion.MSECriterion;
import be.iminds.iot.dianne.nn.train.eval.ArgMaxEvaluator;
import be.iminds.iot.dianne.nn.train.strategy.StochasticGradient;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(service=MNISTDemo.class, 
	property={"osgi.command.scope=mnist",
			"osgi.command.function=train",
			"osgi.command.function=evaluate",
			"osgi.command.function=sample",
			"osgi.command.function=data"},
			immediate=true)
public class MNISTDemo {

	private Input input;
	private Output output;
	private List<Trainable> toTrain = new ArrayList<Trainable>();

	private Dataset dataTrain = null;
	private Dataset dataTest = null;

	private TensorFactory factory = null;
	
	private OutputListener outputLog = new OutputListener() {
		@Override
		public void onForward(Tensor output) {
			System.out.println(output);
		}
	};
	
	private Random rand = new Random(System.currentTimeMillis());
	
	@Reference
	public void setTensorFactory(TensorFactory factory){
		this.factory = factory;
	}
	
	@Reference
	public void setInput(Input input){
		this.input = input;
	}
	
	@Reference
	public void setOutput(Output output){
		this.output = output;
		this.output.addOutputListener(outputLog);
	}

	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addTrainable(Trainable t){
		this.toTrain.add(t);
	}
	
	public void removeTrainable(Trainable t){
		this.toTrain.remove(t);
	}
	
	public void data(String dir){
		System.out.println("Loading MNIST dataset ...");
		this.dataTrain = new MNISTDataset(factory, dir, Set.TRAIN, true);
		this.dataTest = new MNISTDataset(factory, dir, Set.TEST, true);
		System.out.println("Loaded!");
	}
	
	public void train(int batchSize, int noEpochs){
		if(dataTrain==null){
			System.out.println("No dataset loaded");
			return;
		}
		// disable log when training
		this.output.removeOutputListener(outputLog);
		
		System.out.println("Training ...");
		Criterion loss = new MSECriterion(factory);
		Trainer trainer = new StochasticGradient(batchSize, noEpochs);
		trainer.train(input, output, toTrain, loss, dataTrain);
		System.out.println("Trained!");
		
		this.output.addOutputListener(outputLog);
	}
	
	public void evaluate(){
		if(dataTest==null){
			System.out.println("No dataset loaded");
			return;
		}
		
		// disable log when evaluating
		this.output.removeOutputListener(outputLog);
		
		System.out.println("Evaluating...");
		Evaluator eval = new ArgMaxEvaluator(factory);
		Evaluation result = eval.evaluate(input, output, dataTest);
		System.out.println("Accuracy: "+result.accuracy());
		
		this.output.addOutputListener(outputLog);
	}
	
	public void sample(){
		if(dataTest==null){
			System.out.println("No dataset loaded");
			return;
		}
		
		int index = rand.nextInt(dataTest.size());
		Tensor sample = dataTest.getInputSample(index);
		Tensor expected = dataTest.getOutputSample(index);
		System.out.println("Expected: "+expected);
		this.input.input(sample);
	}
}
