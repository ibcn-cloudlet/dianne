package be.iminds.iot.dianne.dataset.cifar10;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.dataset.Dataset;
import be.iminds.iot.dianne.dataset.Sample;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

/**
 * The CIFAR-10 dataset, uses the binary images from:
 * http://www.cs.toronto.edu/~kriz/cifar.html
 * 
 * @author tverbele
 *
 */
@Component(immediate=true)
public class Cifar10Dataset implements Dataset {

	private TensorFactory factory;
	
	private List<Sample> data = new ArrayList<Sample>();
	private String[] labels;
	
	private int noRows;
	private int noColumns;
	private int inputSize;
	private int outputSize;
	private int noSamples;
	
	private String dir = "";
	// thread to start loading data when constructed
	private ExecutorService loader = Executors.newSingleThreadExecutor();
	
	@Reference
	void setTensorFactory(TensorFactory f){
		this.factory = f;
	}
	
	@Activate
	public void activate(BundleContext context){
		String d = context.getProperty("be.iminds.iot.dianne.dataset.cifar10.location");
		if(d!=null){
			this.dir = d;
		}
		
		noRows = 32;
		noColumns = 32;
		inputSize = noRows*noColumns*3;
		outputSize = 10;
		noSamples = 60000;
		
		readLabels("batches.meta.txt");
		// merge all samples into one dataset
		read("data_batch_1.bin");
		read("data_batch_2.bin");
		read("data_batch_3.bin");
		read("data_batch_4.bin");
		read("data_batch_5.bin");
		read("test_batch.bin");
	}
	
	private void readLabels(String file){
		try {
			InputStream labelInput = new FileInputStream(dir+file);
			
			labels = new String[10];
			BufferedReader reader = new BufferedReader(new InputStreamReader(labelInput));
			for(int i=0;i<10;i++){
				labels[i] = reader.readLine();
			}
			reader.close();
			
			System.out.println(Arrays.toString(labels));
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	private void read(String file){
		try {
			InputStream input = new FileInputStream(dir+file);
			
			loader.execute(new Runnable() {
				@Override
				public void run() {
					parse(input, 10000);
				}
			});
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	@Override
	public String getName(){
		return "CIFAR-10";
	}
	
	@Override
	public int size() {
		return noSamples;
	}

	@Override
	public int inputSize() {
		return inputSize;
	}

	@Override
	public int outputSize() {
		return outputSize;
	}

	@Override
	public Tensor getInputSample(int index) {
		// some hack to allow prefecting on construction
		// TODO better solution?
		while(data.size()<=index && index < noSamples){
			synchronized(data){
				if(data.size()<=index ){
					try {
						data.wait();
					} catch (InterruptedException e) {
					}
				}
			}
		}
		return data.get(index).input;
	}

	@Override
	public Tensor getOutputSample(int index) {
		// some hack to allow prefecting on construction
		// TODO better solution?
		while(data.size()<=index && index < noSamples){
			synchronized(data){
				if(data.size()<=index && index < noSamples){
					try {
						data.wait();
					} catch (InterruptedException e) {
					}
				}
			}
		}
		return data.get(index).output;
	}
	
	@Override
	public String[] getLabels(){
		return labels;
	}

	private int readInt(InputStream is) throws IOException{
		byte[] b = new byte[4];
		is.read(b, 0, 4);
		int i = ((0xFF & b[0]) << 24) | ((0xFF & b[1]) << 16) |
	            ((0xFF & b[2]) << 8) | (0xFF & b[3]);
		return i;
	}
	
	private int readUByte(InputStream is) throws IOException{
		byte[] b = new byte[1];
		is.read(b, 0, 1);
		int i = (0xFF & b[0]);
		return i;
	}
	
	private void parse(InputStream input, int count) {
		try {
			for(int read = 0;read<count;read++){
				Tensor out = factory.createTensor(10);
				out.fill(0.0f);
				
				int i = readUByte(input);
				out.set(1.0f, i);
				
				float inputData[] = new float[inputSize];
				for(int j=0;j<inputSize;j++){
					inputData[j] = (float)readUByte(input)/255f;
				}
				Tensor in = factory.createTensor(inputData, 3, noRows, noColumns);
				
				Sample s = new Sample(in, out);
				synchronized(data){
					data.add(s);
					data.notifyAll();
				}
			}
			
		} catch(Exception e){
			e.printStackTrace();
		}
		
	}
	
}
