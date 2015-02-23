package be.iminds.iot.dianne.demo.mnist;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.dataset.Dataset;
import be.iminds.iot.dianne.dataset.DatasetAdapter;
import be.iminds.iot.dianne.nn.module.ForwardListener;
import be.iminds.iot.dianne.nn.module.Input;
import be.iminds.iot.dianne.nn.module.Output;
import be.iminds.iot.dianne.nn.module.Preprocessor;
import be.iminds.iot.dianne.nn.module.Trainable;
import be.iminds.iot.dianne.nn.train.Criterion;
import be.iminds.iot.dianne.nn.train.Evaluation;
import be.iminds.iot.dianne.nn.train.criterion.MSECriterion;
import be.iminds.iot.dianne.nn.train.eval.ArgMaxEvaluator;
import be.iminds.iot.dianne.nn.train.eval.EvalProgressListener;
import be.iminds.iot.dianne.nn.train.strategy.StochasticGradient;
import be.iminds.iot.dianne.nn.train.strategy.TrainProgressListener;
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
	private List<Preprocessor> preprocessors = new ArrayList<Preprocessor>();

	private Dataset dataTrain = null;
	private Dataset dataTest = null;

	private TensorFactory factory = null;
	
	private ForwardListener outputLog = new ForwardListener() {
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
		this.output.addForwardListener(outputLog);
	}

	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addTrainable(Trainable t){
		this.toTrain.add(t);
	}
	
	public void removeTrainable(Trainable t){
		this.toTrain.remove(t);
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addPreprocessor(Preprocessor p){
		this.preprocessors.add(p);
	}
	
	public void removePreprocessor(Preprocessor p){
		this.preprocessors.remove(p);
	}
	
	@Reference
	public void setDataset(Dataset dataset){
		this.dataTrain = new DatasetAdapter(dataset, 0, 60000);
		this.dataTest = new DatasetAdapter(dataset, 60000, 70000);
	}
	
	public void train(int batchSize, int noEpochs){
		if(dataTrain==null){
			System.out.println("No dataset loaded");
			return;
		}
		// disable log when training
		this.output.removeForwardListener(outputLog);
		
		System.out.println("Training ...");
		Criterion loss = new MSECriterion(factory);
		StochasticGradient trainer = new StochasticGradient(batchSize, noEpochs, 0.5f, 0f);
		trainer.addProgressListener(new TrainProgressListener() {
			
			@Override
			public void onProgress(int epoch, int batch, float error) {
				System.out.println(epoch+"\t"+batch+"\t"+error);
			}
		});
		trainer.train(input, output, toTrain, preprocessors, loss, dataTrain);
		System.out.println("Trained!");
		
		this.output.addForwardListener(outputLog);
	}
	
	public void evaluate(){
		if(dataTest==null){
			System.out.println("No dataset loaded");
			return;
		}
		
		// disable log when evaluating
		this.output.removeForwardListener(outputLog);
		
		System.out.println("Evaluating...");
		ArgMaxEvaluator eval = new ArgMaxEvaluator(factory);
		eval.addProgressListener(new EvalProgressListener() {
			@Override
			public void onProgress(Tensor confusionMatrix) {
				System.out.println(confusionMatrix);
			}
		});
		Evaluation result = eval.evaluate(input, output, dataTest);
		System.out.println("Accuracy: "+result.accuracy());
		
		this.output.addForwardListener(outputLog);
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
