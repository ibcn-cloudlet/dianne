package be.iminds.iot.dianne.things.light;

import java.awt.Color;

import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.thing.light.Light;

public class LightOutput implements ForwardListener {

	private final Light light;
	private final TensorFactory factory;
	
	public LightOutput(TensorFactory factory, Light light){
		this.light = light;
		this.factory = factory;
	}
	
	@Override
	public void onForward(Tensor output) {
		if(output.size(0)==1){
			// on/off of lamp
			float v = output.get(0);
			if(v < 0.5f){
				// turn off
				light.off();
			}  else {
				light.on();
			}
		} else if(output.size(0)==2){
			// if argmax == 0 : n, if argmax == 1 : off
			int i = factory.getTensorMath().argmax(output);
			if(i==0){
				light.on();
			} else {
				light.off();
			}
		} else if(output.size(0)==3){
			// interpret as r,g,b colors
			light.setColor(new Color(output.get(0),output.get(1),output.get(2)));
			light.on();
		} else {
			// generate color hues based on size and argmax
			float stride = 1.0f/output.size(0);
			int i = factory.getTensorMath().argmax(output);
			Color c = getColor(i*stride);
			light.setColor(c);
			light.on();
		}
	}

	
	
	// generate rgb color from just hue (0..1) (set saturation and value to max)
	private static Color getColor(float hue){
		float value = 1.0f;
		float saturation = 1.0f;
		float c = value*saturation;
		
		float h = hue*6;   // when hue 0..360 then h = hue/60 thus ....
		float x = c*(1 - Math.abs(h % 2 -1));
		
		float r = 0;
		float g = 0;
		float b = 0;
		
		if(h < 1.0) {
			r = c;
			g = x;
		} else if(h < 2.0) {
			r = x;
			g = c;
		} else if(h < 3.0) {
			g = c;
			b = x;
		} else if(h < 4.0) {
			g = x;
			b = c;
		} else if(h < 5.0) {
			r = x;
			b = c;
		} else if(h <= 6.0) {
			r = c;
			b = x;
		}
	
		float min = value - c;
		 
		r += min;
		g += min;
		b += min;
		
		Color rgb = new Color(r, g, b);
		return rgb;
	}
	
}
