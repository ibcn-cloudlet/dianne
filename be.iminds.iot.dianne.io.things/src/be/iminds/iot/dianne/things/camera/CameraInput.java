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
	private int width;
	private int height;
	private int channels;
	
	public CameraInput(TensorFactory factory, Input input,
			int width, int height, int channels){
		this.factory = factory;
		this.input = input;
		
		this.width = width;
		this.height = height;
		this.channels = channels;
		
		this.buffer = new float[channels*width*height];
	}
	
	
	@Override
	public void nextFrame(UUID id, Format format, byte[] data) {
		int k = 0;
		for(int c=0;c<channels;c++){
			for(int y=0;y<height;y++){
				for(int x=0;x<width;x++){
					float val = (data[c*width*height+y*width+x] & 0xFF)/255f;
					buffer[k++] = val;
				}
			}
		}
		Tensor in = factory.createTensor(buffer, channels, height, width);
		input.input(in);
	}


	
}
