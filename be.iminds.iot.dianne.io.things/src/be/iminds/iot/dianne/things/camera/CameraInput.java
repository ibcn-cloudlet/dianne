package be.iminds.iot.dianne.things.camera;

import java.awt.BorderLayout;
import java.awt.Canvas;
import java.awt.Frame;
import java.awt.Graphics2D;
import java.awt.Toolkit;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferStrategy;
import java.awt.image.BufferedImage;

import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.thing.camera.CameraListener;

public class CameraInput extends Canvas implements CameraListener {

	private final TensorFactory factory;
	private final Input input;
	
	private float[] buffer;
	
	private CameraCanvas canvas;
	
	public CameraInput(TensorFactory factory, Input input){
		this.factory = factory;
		this.input = input;
		this.buffer = new float[28*28];
		
		canvas = new CameraCanvas();
	}
	
	@Override
	public void nextFrame(byte[] data) {
		// TODO should have a auto Reshape module
		// for now fixed coded for MNIST
		
		// take a window of 224x244 and subsample by 8
		int stride = 320;
		
		int startX = (320-224)/2;
		int endX = startX+224;
		
		int startY = (240-224)/2;
		int endY = startY+224;
		

		int k = 0;
		for(int y=startY;y<endY;y+=8){
			for(int x=startX;x<endX;x+=8){
				buffer[k++] = (data[y*stride+x] & 0xFF)/255f;
			}
		}
		Tensor in = factory.createTensor(buffer, 28,28);
		input.input(in);
		
		canvas.render(in);
	}

	

	
}
