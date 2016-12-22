/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  iMinds - IBCN - UGent
 *
 * This file is part of DIANNE.
 *
 * DIANNE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.things.input;

import java.util.Dictionary;
import java.util.Hashtable;
import java.util.UUID;

import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceRegistration;

import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.util.ImageConverter;
import be.iminds.iot.things.api.camera.Camera;
import be.iminds.iot.things.api.camera.Camera.Format;
import be.iminds.iot.things.api.camera.CameraListener;

public class CameraInput extends ThingInput implements CameraListener {

	private final Camera camera;
	
	private ImageConverter converter;
	private float[] buffer;

	// input frame
	private int width;
	private int height;
	private int channels;

	private ServiceRegistration registration;
	
	public CameraInput(UUID id, String name, Camera camera){
		super(id, name, "Camera");
		
		this.camera = camera;
	}
	
	
	@Override
	public void nextFrame(UUID id, Format format, byte[] data) {
		Tensor in = null;
		if(format==Format.MJPEG){
			try {
				in = converter.fromBytes(data);
			} catch(Exception e){
				System.out.println("Failed to read jpeg frame");
			}
		} else if(format==Format.RGB){
			int k = 0;
			for(int c=0;c<channels;c++){
				for(int y=0;y<height;y++){
					for(int x=0;x<width;x++){
						float val = (data[c*width*height+y*width+x] & 0xFF)/255f;
						buffer[k++] = val;
					}
				}
			}
			in = new Tensor(buffer, channels, height, width);
		} 

		if(in != null){
			for(Input input: inputs){
				input.input(in);
			}
		}
	}


	@Override
	public void connect(Input input, BundleContext context) {
		super.connect(input, context);

		if(registration == null){
			try {
				camera.setFramerate(15f);
				camera.start(320, 240, Camera.Format.MJPEG);
			} catch(Exception e){
				System.err.println("Error starting camera");
			}
			
			this.width = 320;
			this.height = 240;
			this.channels = 3;
	
			this.converter = new ImageConverter();
			this.buffer = new float[channels*width*height];
			
			Dictionary<String, Object> properties = new Hashtable<String, Object>();
			properties.put(CameraListener.CAMERA_ID, id.toString());
			properties.put("aiolos.unique", true);
			registration = context.registerService(CameraListener.class.getName(), this, properties);
		}
	}

	
	@Override
	public void disconnect(Input input){
		super.disconnect(input);
		
		if(inputs.size() == 0){
			try {
				camera.stop();
			} catch(Exception e){}
			
			if(registration != null){
				registration.unregister();
				registration = null;
			}
		}
	}
}
