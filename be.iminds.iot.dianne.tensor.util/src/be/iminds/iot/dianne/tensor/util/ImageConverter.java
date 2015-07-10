package be.iminds.iot.dianne.tensor.util;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class ImageConverter {

	private final TensorFactory factory;
	
	public ImageConverter(TensorFactory f){
		this.factory = f;
	}

	public Tensor readFromImage(BufferedImage img){
		// TODO also support other image formats?
		int width = img.getWidth();
		int height = img.getHeight();
		
		float[] imageData = new float[width*height*3];
		
		float[] rgb = new float[3];
		int r = 0;
		int g = width*height;
		int b = 2*width*height;
		for(int j=0;j<height;j++){
			for(int i=0;i<width;i++){
				rgb = img.getRaster().getPixel(i, j, rgb);
				imageData[r++] = rgb[0]/255f;
				imageData[g++] = rgb[1]/255f;
				imageData[b++] = rgb[2]/255f;
			}
		}
		
		return factory.createTensor(imageData, 3, height, width);
	}
	
	public Tensor readFromFile(String fileName) throws IOException{
		BufferedImage img = ImageIO.read(new File(fileName));
		return readFromImage(img);
	}

	public BufferedImage writeToImage(Tensor t) throws Exception {
		int width, height, channels;
		
		if(t.dim()==3){
			channels = t.dims()[0];
			height = t.dims()[1];
			width = t.dims()[2];
		} else if(t.dim()==2){
			channels = 1;
			height = t.dims()[0];
			width = t.dims()[1];
		} else {
			throw new Exception("Wrong dimensions of tensor");
		}
		
		BufferedImage img = new BufferedImage(width, height,
				BufferedImage.TYPE_INT_RGB);
		
		float[] data = t.get();
		int c1 = 0;
		int c2 = width*height;
		int c3 = 2*width*height;
		
		int r=0,g=0,b=0,a=0,col;
		
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				if(channels==1){
					int val = (int)(data[c1++]*255f);
					r = val;
					g = val;
					b = val;
					a = 255;
				} else if(channels==3){
					r = (int)(data[c1++]*255f);
					g = (int)(data[c2++]*255f);
					b = (int)(data[c3++]*255f);
					a = 255;
				}

				col = a << 24 | r << 16 | g << 8 | b;
				img.setRGB(i, j, col);
			}
		}
		return img;
	}
	
	public void writeToFile(String fileName, Tensor t) throws Exception{
		BufferedImage img = writeToImage(t);
		String formatName = fileName.substring(fileName.length()-3);
		ImageIO.write(img, formatName, new File(fileName));
	}
}
