package be.iminds.iot.dianne.dataset.util;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.Transparency;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class ImageLoader {

	private TensorFactory factory;
	
	public ImageLoader(TensorFactory f){
		this.factory = f;
	}
	
	/**
	 * Loads an image from file as a 3D RGB tensor with size x , y
	 */
	public Tensor loadImageRGB(String file, int x, int y){
		float[] imageData = new float[3*x*y];
		try {
			BufferedImage img = ImageIO.read(new File(file));
			// scale the smallest size to whished size and
			// crop the central part of the other to get a square image patch
			
			int xOffset = 0;
			int yOffset = 0;
			float factor = 1.0f;
			
			float factorX = (float) (x +1)/ img.getWidth();
			float factorY = (float) (y +1)/ img.getHeight();
			if (factorX <= factorY) {
				factor = factorY;
			} else {
				factor = factorX;
			}
			
			xOffset = (int)(img.getWidth()*factor - x)/2;
			yOffset = (int)(img.getHeight()*factor - y)/2;

			BufferedImage scaled = getScaledInstance(img, factor);
			BufferedImage cropped = scaled.getSubimage(xOffset, yOffset, x, y);
		
			float[] rgb = new float[3];
			int a = 0;
			int b = x*y;
			int c = 2*x*y;
			for(int j=0;j<y;j++){
				for(int i=0;i<x;i++){
					rgb = cropped.getRaster().getPixel(i, j, rgb);
					imageData[a++] = rgb[0]/255f;
					imageData[b++] = rgb[1]/255f;
					imageData[c++] = rgb[2]/255f;
				}
			}
			
			// TODO also support multi-scale classification?
		} catch (IOException e) {
			e.printStackTrace();
		}
		return factory.createTensor(imageData, 3, y, x);
	}
	
	// scaling method from https://today.java.net/article/2007/03/30/perils-imagegetscaledinstance
	private BufferedImage getScaledInstance(BufferedImage img, float factor) {
		int type = (img.getTransparency() == Transparency.OPAQUE) ? BufferedImage.TYPE_INT_RGB
				: BufferedImage.TYPE_INT_ARGB;
		BufferedImage ret = (BufferedImage) img;

		int w = (int)(img.getWidth()*factor);
		int h = (int)(img.getHeight()*factor);

		BufferedImage tmp = new BufferedImage(w, h, type);
		Graphics2D g2 = tmp.createGraphics();
		g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
				RenderingHints.VALUE_INTERPOLATION_BILINEAR);
		g2.drawImage(ret, 0, 0, w, h, null);
		g2.dispose();

		ret = tmp;

		return ret;
	}
}
