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
	
	// input frame
	private int frameWidth;
	private int frameHeight;
	private int frameChannels;
	
	// target - for now hard coded for the MNIST NN
	private int channels = 1;
	private int width = 28;
	private int height = 28;
	
	private int subsample = 4;
	private boolean threshold = true;
	
	public CameraInput(TensorFactory factory, Input input,
			int frameWidth, int frameHeight, int frameChannels){
		this.factory = factory;
		this.input = input;
		
		this.frameWidth = frameWidth;
		this.frameHeight = frameHeight;
		this.frameChannels = frameChannels;
		
		if(frameChannels==3){
			channels = 3;
		}
		
		this.buffer = new float[channels*width*height];
	}
	
	
	@Override
	public void nextFrame(UUID id, Format format, byte[] data) {
		// TODO should have a auto Reshape module
		// for now fixed coded for MNIST
		int startX = (frameWidth-width*subsample)/2;
		int endX = startX+width*subsample;
		
		int startY = (frameHeight-height*subsample)/2;
		int endY = startY+height*subsample;
		

		int k = 0;
		for(int c=0;c<channels;c++){
			for(int y=startY;y<endY;y+=subsample){
				for(int x=startX;x<endX;x+=subsample){
					float val = (data[c*frameWidth*frameHeight+y*frameWidth+x] & 0xFF)/255f;
					if(threshold){
						buffer[k++] = val > 0.5f ? 0.0f : 1.0f;
					} else {
						buffer[k++] = val;
					}
				}
			}
		}
		Tensor in; 
		if(channels==1){
			in = factory.createTensor(buffer, height, width);
		} else {
			in = factory.createTensor(buffer, channels, height, width);
		}
		input.input(in);
	}


	
}
