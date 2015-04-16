package be.iminds.iot.dianne.dataset.imagenet;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.Transparency;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

/**
 * Sample of the Imagenet dataset
 * 
 * Currently based on the ILSVRC 2012 validation set
 * 
 * @author tverbele
 *
 */
@Component(immediate = true, property = { "aiolos.callback=be.iminds.iot.dianne.api.dataset.Dataset" })
public class ImageNetDataset implements Dataset {

	private TensorFactory factory;

	private List<Sample> data = new ArrayList<Sample>();
	private String[] labels;
	private int[] outputs;

	private int noRows;
	private int noColumns;
	private int inputSize;
	private int outputSize;
	private int noSamples;

	private String dir = "";

	@Reference
	void setTensorFactory(TensorFactory f) {
		this.factory = f;
	}

	@Activate
	public void activate(BundleContext context) {
		String d = context
				.getProperty("be.iminds.iot.dianne.dataset.imagenet.location");
		if (d != null) {
			this.dir = d;
		}

		outputSize = 1000;

		// TODO this is configurable here, scaled from JPEG?
		noRows = 231;
		noColumns = 231;
		inputSize = noRows * noColumns * 3;

		readLabels("classes.txt");
		readOutputs("outputs.txt");

		noSamples = outputs.length;
	}

	private void readLabels(String file) {
		try {
			InputStream labelInput = new FileInputStream(dir + file);

			ArrayList<String> l = new ArrayList<String>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					labelInput));
			String s;
			while ((s = reader.readLine()) != null) {
				// only pick first label in case of multiple definitions
				int comma = s.indexOf(",");
				if (comma > 0) {
					s = s.substring(0, comma);
				}
				l.add(s);
			}
			labels = new String[l.size()];
			for (int i = 0; i < l.size(); i++) {
				labels[i] = l.get(i);
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void readOutputs(String file) {
		try {
			InputStream outputsInput = new FileInputStream(dir + file);

			ArrayList<Integer> l = new ArrayList<Integer>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					outputsInput));
			String s;
			while ((s = reader.readLine()) != null) {
				// only pick first label in case of multiple definitions
				int clazz = Integer.parseInt(s) - 1; // in the file this ranges
														// from 1..1000, convert
														// to 0..999
				l.add(clazz);
			}
			outputs = new int[l.size()];
			for (int i = 0; i < l.size(); i++) {
				outputs[i] = l.get(i);
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public String getName() {
		return "ImageNet";
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
		// Open JPEG file and convert to size
		String file = dir + "images/" + "ILSVRC2012_val_"
				+ String.format("%08d", index) + ".JPEG";
		System.out.println("Read file " + file);
		float[] imageData = new float[inputSize];
		try {
			BufferedImage img = ImageIO.read(new File(file));
			// scale the smallest size to whished size and
			// crop the central part of the other to get a square image patch
			
			float factor = 1.0f;
			int xOffset = 0;
			int yOffset = 0;
			if (img.getWidth() <= img.getHeight()) {
				factor = (float) noColumns / img.getWidth();
				yOffset = (int)(img.getHeight()*factor - noRows)/2;
			} else {
				factor = (float) noRows / img.getHeight();
				xOffset = (int)(img.getWidth()*factor - noColumns)/2;
			}

			BufferedImage scaled = getScaledInstance(img, factor);
			BufferedImage cropped = scaled.getSubimage(xOffset, yOffset, noColumns, noRows);
			
		
			float[] rgb = new float[3];
			int a = 0;
			int b = noColumns*noRows;
			int c = 2*noColumns*noRows;
			for(int j=0;j<noRows;j++){
				for(int i=0;i<noColumns;i++){
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
		return factory.createTensor(imageData, 3, noRows, noColumns);
	}

	@Override
	public Tensor getOutputSample(int index) {
		Tensor output = factory.createTensor(outputSize);
		output.fill(0.0f);
		output.set(1.0f, outputs[index]);
		return output;
	}

	@Override
	public String[] getLabels() {
		return labels;
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
