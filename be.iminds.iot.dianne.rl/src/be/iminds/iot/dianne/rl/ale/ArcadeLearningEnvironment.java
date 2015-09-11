package be.iminds.iot.dianne.rl.ale;

import java.io.File;
import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.rl.Environment;
import be.iminds.iot.dianne.api.rl.EnvironmentListener;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

/**
 * Arcade Learning Environment for learning agents to play Atari games
 *  
 * @author tverbele
 *
 */
@Component(immediate = true,
property = { "name="+ArcadeLearningEnvironment.NAME, "aiolos.unique=be.iminds.iot.dianne.api.rl.Environment" })
public class ArcadeLearningEnvironment implements Environment {

	public static final String NAME = "ALE";
	
    static {
		try {
		    System.loadLibrary("ALE");
		} catch (final UnsatisfiedLinkError e) {
		    System.err.println("Native code library ALE failed to load. \n"+ e);
		    throw e;
		}
    }

    private TensorFactory factory;
	private Set<EnvironmentListener> listeners = Collections.synchronizedSet(new HashSet<>());

	private String rom = "roms/pong.bin"; 
	private int skip = 1;
	private Tensor observation;
    
    @Activate
    public void activate(BundleContext context) throws Exception {
    	String r = context.getProperty("be.iminds.iot.dianne.rl.ale.rom");
    	if(r!=null){
    		rom = r;
    	}
    	
		String sk = context.getProperty("be.iminds.iot.dianne.rl.ale.skip");
		if (sk != null)
			this.skip = Integer.parseInt(sk);
    	
    	// check if file exists
    	File f = new File(rom);
    	if(!f.exists()){
    		System.err.println("Failed to initialize ALE - ROM "+rom+" does not exist!");
    		throw new Exception("ROM "+rom+" does not exist!");
    	}
    	
    	loadROM(rom);
    	setFrameskip(skip);
    	
    	System.out.println("Loaded rom "+rom+", this has "+getActions()+" valid actions");
    	
    	observation = factory.createTensor(getScreen(), 3, 210, 160);
    }
    
	@Override
	public float performAction(Tensor action) {
		int r = performAction(factory.getTensorMath().argmax(action));
		
		final float reward = r;
    	observation = factory.createTensor(getScreen(), 3, 210, 160);
		
		synchronized(listeners){
			listeners.stream().forEach(l -> l.onAction(reward, observation));
		}
    	
		return r > 0 ? 1 : r < 0 ? -1 : 0;
	}

	@Override
	public Tensor getObservation() {
		if(gameOver()){
			return null;
		} else {
			return observation;
		}
	}

	@Override
	public void reset() {
		resetGame();
    	observation = factory.createTensor(getScreen(), 3, 210, 160);
	}

	private native void loadROM(String rom);

	private native int getActions();
	
	private native int performAction(int index);
	
	private native void resetGame();
	
	private native boolean gameOver();
	
	private native float[] getScreen();
	
	private native void setFrameskip(int skip);
	
	@Reference
	void setTensorFactory(TensorFactory factory) {
		this.factory = factory;
	}
	
	@Reference(cardinality = ReferenceCardinality.MULTIPLE, policy = ReferencePolicy.DYNAMIC)
	void addEnvironmentListener(EnvironmentListener l, Map<String, Object> properties){
		String target = (String) properties.get("target");
		if(target == null || target.equals(NAME)){
			listeners.add(l);
		}
	}
	
	void removeEnvironmentListener(EnvironmentListener l){
		listeners.remove(l);
	}
}
