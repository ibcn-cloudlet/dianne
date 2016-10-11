package be.iminds.iot.dianne.nn.learn.strategy;

import be.iminds.iot.dianne.api.nn.learn.LearnProgress;

public class GenerativeAdverserialLearnProgress extends LearnProgress {

	public final float d_loss;
	public final float g_loss;
	
	public GenerativeAdverserialLearnProgress(long iteration, 
			float d_loss_pos, float d_loss_neg,
			float g_loss) {
		super(iteration, g_loss);
		this.d_loss = d_loss_pos+d_loss_neg;
		this.g_loss = g_loss;
	}

	@Override
	public String toString(){
		return "[LEARNER] Batch: "+iteration+" D Loss: "+d_loss+" G Loss: "+g_loss;
	}
}
