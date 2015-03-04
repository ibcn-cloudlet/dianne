package be.iminds.iot.dianne.dataset.stl10;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

/**
 * The CIFAR-10 dataset, uses the binary images from:
 * http://www.cs.toronto.edu/~kriz/cifar.html
 * 
 * @author tverbele
 *
 */
@Component(immediate=true, property={"aiolos.callback=be.iminds.iot.dianne.dataset.Dataset"})
public class STL10Dataset implements Dataset {

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
		String d = context.getProperty("be.iminds.iot.dianne.dataset.stl10.location");
		if(d!=null){
			this.dir = d;
		}
		
		noRows = 96;
		noColumns = 96;
		inputSize = noRows*noColumns*3;
		outputSize = 10;
		noSamples = 13000;
		
		readLabels("class_names.txt");
		// merge all samples into one dataset
		read("train");
		read("test");
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
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	private void read(String file){
		try {
			InputStream imageInput = new FileInputStream(dir+file+"_X.bin");
			InputStream labelInput = new FileInputStream(dir+file+"_y.bin");
			
			loader.execute(new Runnable() {
				@Override
				public void run() {
					parse(imageInput, labelInput);
				}
			});
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	@Override
	public String getName(){
		return "STL-10";
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
	
	private void parse(InputStream imageInput, InputStream labelInput) {
		try {
			while(imageInput.available()>0
					&& labelInput.available()>0){
				Tensor out = factory.createTensor(10);
				out.fill(0.0f);
				
				int i = readUByte(labelInput);
				// categories are from 1..10
				out.set(1.0f, i-1);
				
				float inputData[] = new float[inputSize];
				// STL10 is formatted column-major, convert to row-major
				for(int c=0;c<3;c++){
					for(int y=0;y<96;y++){
						for(int x=0;x<96;x++){
							inputData[c*96*96+x*96+y] = (float)readUByte(imageInput)/255f;
						}
					}
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
