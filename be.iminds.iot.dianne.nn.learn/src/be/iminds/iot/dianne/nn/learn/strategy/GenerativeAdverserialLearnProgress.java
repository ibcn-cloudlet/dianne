package be.iminds.iot.dianne.nn.learn.strategy;

import be.iminds.iot.dianne.api.nn.learn.LearnProgress;

public class GenerativeAdverserialLearnProgress extends LearnProgress {

	public final float d_error_positive;
	public final float d_error_negative;
	public final float g_error;
	
	public GenerativeAdverserialLearnProgress(long iteration, 
			float d_error_positive, float d_error_negative,
			float g_error) {
		super(iteration, g_error);
		this.d_error_negative = d_error_negative;
		this.d_error_positive = d_error_positive;
		this.g_error = g_error;
	}

	@Override
	public String toString(){
		return "[LEARNER] Batch: "+iteration+" D+ Error: "+d_error_positive+" D- Error: "+d_error_negative+" G Error: "+g_error;
	}
}
