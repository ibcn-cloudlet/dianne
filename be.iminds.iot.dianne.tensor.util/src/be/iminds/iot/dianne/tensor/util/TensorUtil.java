package be.iminds.iot.dianne.tensor.util;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public abstract class TensorUtil {
	
	public static Tensor fromFile(String fileName) throws FileNotFoundException, IOException {
		Tensor t = null;
		
		File f = new File(fileName);
		try (DataInputStream is = 
				new DataInputStream(
						new BufferedInputStream(
								new FileInputStream(f)))){
			// load tensor in chunks, slightly slower than one copy from Java to native,
			// but reduces memory usage a lot for big tensors
			int bufferSize = 10000;
			float[] data = new float[bufferSize];
			
			int noDims = is.readInt();
			System.out.println("dims lenght: " + noDims);
			int[] dims = new int[noDims];
			for (int i=0; i<noDims; i++) {
				dims[i] = is.readInt();
				System.out.println("dim: " + dims[i]);
			}
			
			t = new Tensor(dims);
			int length = t.size();
			t.reshape(length); // reshape to allow narrowing on entire data
			System.out.println("Size: " + t.size() + "; data size: " + length);
			
			int index = 0;
			while(length > 0){
				if(length<bufferSize){
					bufferSize = length;
					data = new float[bufferSize];
				}
				
				for(int i=0;i<bufferSize;i++){
					data[i] = is.readFloat();
				}
				
				System.out.println("data size: " + index + "/" + (index+bufferSize-1) + "/" + (t.size()-1));
				t.narrow(0, index, bufferSize).set(data);				
				
				length -= bufferSize;
				index+= bufferSize;
			}
			t.reshape(dims); // reshape to original shape
			is.close();
		}
		return t;
	}
	
	public static void toFile(String fileName, Tensor t) throws FileNotFoundException, IOException {
		File f = new File(fileName);
		try(DataOutputStream os = 
				new DataOutputStream(
						new BufferedOutputStream(
								new FileOutputStream(f)))) {
			os.writeInt(t.dims().length);
			System.out.println("dims lenght: " + t.dims().length);
			for (int dim : t.dims()) {
				os.writeInt(dim);
				System.out.println("dim: " + dim);
			}
			float[] data = t.get();
			System.out.println("Size: " + t.size() + "; data size: " + data.length);
			
			for(float value : data){
				os.writeFloat(value);
			}
			os.flush();
			os.close();
		}
	}
}
