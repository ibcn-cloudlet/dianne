package be.iminds.iot.dianne.dataset.mnist;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import be.iminds.iot.dianne.dataset.Dataset;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class MNISTDataset implements Dataset{

	private final TensorFactory factory;
	
	//private Tensor data;
	List<Sample> data = new ArrayList<Sample>();
	
	private int inputSize;
	private int outputSize;
	private int noSamples;
	
	private String dir = "";
	
	private class Sample {
		final Tensor input;
		final Tensor output;
		
		public Sample(Tensor i, Tensor o){
			this.input = i;
			this.output = o;
		}
	}
	
	public MNISTDataset(TensorFactory factory, String dir) {
		this.factory = factory;
		this.dir = dir;

		// merge train and test samples into one dataset
		read("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
		read("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
	}
	
	public void read(String images, String labels){
		try {
			InputStream imageInput = new FileInputStream(dir+images);

			int magic = readInt(imageInput);
			assert magic == 2051;
			int noImages = readInt(imageInput);
			int noRows = readInt(imageInput);
			int noColumns = readInt(imageInput);
			
			InputStream labelInput = new FileInputStream(dir+labels);
			magic = readInt(labelInput);
			assert magic == 2049;
			int noLabels = readInt(labelInput);

			System.out.println("Reading MNIST dataset");
			System.out.println("#Images: "+noImages+" #Rows: "+noRows+" #Columns: "+noColumns+" #Labels: "+noLabels);

			assert noLabels == noImages;
			
			noSamples += noImages;
			inputSize = noRows*noColumns;
			outputSize = 10;
			
			parse(imageInput, labelInput, noImages);
		} catch(IOException e){
			e.printStackTrace();
		}
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
		return data.get(index).input;
	}

	@Override
	public Tensor getOutputSample(int index) {
		return data.get(index).output;
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
	
	private void parse(InputStream imageInput, InputStream labelInput, int count) {
		try {
			for(int read = 0;read<count;read++){
				Tensor input = factory.createTensor(inputSize);
				Tensor output = factory.createTensor(outputSize);
				output.fill(0.0f);
				
				for(int j=0;j<inputSize;j++){
					input.set((float)readUByte(imageInput)/255f,j);
				}
				int i = readUByte(labelInput);
				output.set(1.0f, i);
				
				Sample s = new Sample(input, output);
				data.add(s);
			}
			
		} catch(Exception e){
			e.printStackTrace();
		}
		
	}
}
