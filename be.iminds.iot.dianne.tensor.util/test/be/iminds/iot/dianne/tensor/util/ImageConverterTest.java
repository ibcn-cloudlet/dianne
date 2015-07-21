package be.iminds.iot.dianne.tensor.util;

import org.junit.Test;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.dianne.tensor.impl.java.JavaTensorFactory;

public class ImageConverterTest {
	
	private TensorFactory factory = new JavaTensorFactory();

	@Test
	public void testImageNetImages() {
		ImageConverter conv = new ImageConverter(factory);
		long t1 = System.currentTimeMillis();
		int start = 0;
		int n = 50000;
		for(int i=start;i<start+n;i++){
			String dir = "../tools/datasets/ImageNet/";
			String file = dir + "images/" + "ILSVRC2012_val_"
					+ String.format("%08d", i+1) + ".JPEG";
			try {
				conv.readFromFile(file);
			} catch(Exception e){
				System.out.println("Error with image "+file);
				e.printStackTrace();
			}
		}
		long t2 = System.currentTimeMillis();
		System.out.println("Avg read time: "+(t2-t1)/n+" ms");
	}
	
	@Test
	public void testReadWriteImage() throws Exception {
		ImageConverter conv = new ImageConverter(factory);

		int i = 0;
		String dir = "../tools/datasets/ImageNet/";
		String file = dir + "images/" + "ILSVRC2012_val_"
				+ String.format("%08d", i+1) + ".JPEG";
	
		Tensor t = conv.readFromFile(file);
		conv.writeToFile("test.jpg", t);
	}
}
