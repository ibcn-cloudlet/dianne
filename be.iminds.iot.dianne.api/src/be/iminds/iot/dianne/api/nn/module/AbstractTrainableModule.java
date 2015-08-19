package be.iminds.iot.dianne.api.nn.module;

import java.util.UUID;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

/**
 * Provides base functionality for trainable neural network Modules. Extend this class
 * for creating your own trainable module with one previous and one next Module.
 * 
 * @author tverbele
 *
 */
public abstract class AbstractTrainableModule extends AbstractModule implements Trainable {

	protected Tensor parameters;
	protected Tensor deltaParameters;
	protected boolean fixed = false;
	
	public AbstractTrainableModule(TensorFactory factory) {
		super(factory);
	}
	
	public AbstractTrainableModule(TensorFactory factory, UUID id) {
		super(factory, id);
	}
	
	@Override
	public abstract void accGradParameters();

	@Override
	public void zeroDeltaParameters() {
		deltaParameters.fill(0.0f);
	}

	@Override
	public void updateParameters() {
		if(!fixed){
			factory.getTensorMath().add(parameters, parameters, deltaParameters);
		}
	}
	
	@Override
	public void updateParameters(float scale) {
		if(!fixed){
			factory.getTensorMath().add(parameters, parameters, scale, deltaParameters);
		}
	}

	@Override
	public Tensor getDeltaParameters(){
		return deltaParameters;
	}
	
	@Override
	public Tensor getParameters(){
		return parameters;
	}
	
	@Override
	public void setParameters(Tensor params){
		if(parameters==null){
			parameters = factory.createTensor(params.dims());
		}
		params.copyInto(parameters);
	}
}
