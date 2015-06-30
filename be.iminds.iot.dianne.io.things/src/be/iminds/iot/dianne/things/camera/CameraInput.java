package be.iminds.iot.dianne.things.camera;

import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.things.api.camera.Camera.Format;
import be.iminds.iot.things.api.camera.CameraListener;

public class CameraInput implements CameraListener {

	private final TensorFactory factory;
	private final Input input;
	
	private float[] buffer;
	
	public CameraInput(TensorFactory factory, Input input){
		this.factory = factory;
		this.input = input;
		this.buffer = new float[28*28];
	}
	
	
	@Override
	public void nextFrame(UUID id, Format format, byte[] data) {
		// TODO should have a auto Reshape module
		// for now fixed coded for MNIST
		
		// take a window of 224x244 and subsample by 8
		int w = 28;
		int h = 28;
		int subsample = 4;
		
		int stride = 320;
		
		int startX = (320-w*subsample)/2;
		int endX = startX+w*subsample;
		
		int startY = (240-h*subsample)/2;
		int endY = startY+h*subsample;
		

		int k = 0;
		for(int y=startY;y<endY;y+=subsample){
			for(int x=startX;x<endX;x+=subsample){
				float val = (data[y*stride+x] & 0xFF)/255f;
				buffer[k++] = val > 0.5f ? 0.0f : 1.0f;
			}
		}
		Tensor in = factory.createTensor(buffer, w, h);
		input.input(in);
	}


	
}
