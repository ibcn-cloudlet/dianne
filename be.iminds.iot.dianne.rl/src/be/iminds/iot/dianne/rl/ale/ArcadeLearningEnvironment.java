package be.iminds.iot.dianne.rl.ale;

import java.awt.image.ImageConsumer;
import java.util.Random;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.rl.Environment;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.dianne.tensor.impl.java.JavaTensorFactory;
import be.iminds.iot.dianne.tensor.util.ImageConverter;

/**
 * Arcade Learning Environment for learning agents to play Atari games
 *  
 * @author tverbele
 *
 */
@Component(immediate = true,
property = { "name=ALE", "aiolos.unique=be.iminds.iot.dianne.api.rl.Environment" })
public class ArcadeLearningEnvironment implements Environment {

    static {
		try {
		    System.loadLibrary("ALE");
		} catch (final UnsatisfiedLinkError e) {
		    System.err.println("Native code library ALE failed to load. \n"+ e);
		    throw e;
		}
    }

    private TensorFactory factory;
    
	@Override
	public float performAction(Tensor action) {
		int r = performAction(factory.getTensorMath().argmax(action));
		return r > 0 ? 1 : r < 0 ? -1 : 0;
	}

	@Override
	public Tensor getObservation() {
		float[] screenData = getScreen();
		if(screenData==null){
			return null;
		} else {
			return factory.createTensor(screenData, 3, 210, 160);
		}
	}

	@Override
	public void reset() {
		resetGame();
	}

	private native void loadROM(String rom);

	private native int getActions();
	
	private native int performAction(int index);
	
	private native void resetGame();
	
	private native float[] getScreen();
	
	@Reference
	void setTensorFactory(TensorFactory factory) {
		this.factory = factory;
	}
	
	
	public static void main(String[] args){
		TensorFactory f = new JavaTensorFactory();
		ImageConverter conv = new ImageConverter(f);
		
		ArcadeLearningEnvironment ale = new ArcadeLearningEnvironment();
		ale.setTensorFactory(f);
		
		ale.loadROM("roms/pong.bin");

		System.out.println(ale.getActions());

		
		Random r = new Random();
		int reward = 0;
		int i=0;
		Tensor t;
		while((t = ale.getObservation())!=null){
			try {
				if(i++ % 1000 == 0)
					conv.writeToFile((i++)+".jpg", t);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			reward += ale.performAction(r.nextInt(18));
			System.out.println(reward);
		}
		
		
	}
}
