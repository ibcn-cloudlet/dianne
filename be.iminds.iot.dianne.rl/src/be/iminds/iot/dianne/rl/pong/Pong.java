package be.iminds.iot.dianne.rl.pong;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.rl.Environment;
import be.iminds.iot.dianne.rl.pong.api.PongEnvironment;
import be.iminds.iot.dianne.rl.pong.api.PongListener;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

/**
 * Simple Pong environment in which an agents plays against an AI trying to
 * match the vertical position of the ball.
 * 
 * @author smbohez
 *
 */
@Component(immediate = true,
	property = { "name=Pong", "aiolos.callback=be.iminds.iot.dianne.api.rl.Environment" })
public class Pong implements PongEnvironment, Environment {

	private TensorFactory factory;

	private Set<PongListener> listeners = Collections.synchronizedSet(new HashSet<>());
	
	// paddle length and width
	private float pl = 0.3f;  
	private float pw = 0.05f; 
	// bounds
	private float b = 0.8f;
	// ball radius
	private float rad = 0.025f;
	// speed
	private float vdef = 0.02f; 
	// speedup when bouncing
	private float m = 1.5f;
	
	private float x, y, vx, vy, p, o;

	@Activate
	void activate(BundleContext context) {
		String l = context.getProperty("be.iminds.iot.dianne.rl.pong.paddlelength");
		if (l != null)
			this.pl = Float.parseFloat(l);

		String vdef = context.getProperty("be.iminds.iot.dianne.rl.pong.defaultspeed");
		if (vdef != null)
			this.vdef = Float.parseFloat(vdef);

		reset();
	}

	@Deactivate
	void deactivate() {

	}

	@Override
	public float performAction(Tensor action) {
		float d_p = vdef * ((action.get(0) > 0) ? 1 : (action.get(1) > 0) ? 0 : -1);
		float d_o = vdef * selectOpponentAction();

		p += d_p;
		o += d_o;

		p = Math.min(1*b - pl / 2, p);
		p = Math.max(pl / 2 - 1*b, p);

		o = Math.min(1*b - pl / 2, o);
		o = Math.max(pl / 2 - 1*b, o);

		x += vx;
		y += vy;

		if (y - rad < -1 * b) {
			y = -b + rad;
			vy = -vy;
		} else if (y + rad > 1 * b) {
			y = b - rad;
			vy = -vy;
		}

		float r = 0;

		if (x - rad - pw < -1) {
			if(onPaddle(p)){
				float i = (y - o) * 2 / pl;
				double a = Math.PI / 4 * i;
				float v = vdef + i * i * m * vdef;
				vx = v * (float) Math.cos(a);
				vy = v * (float) Math.sin(a);
				x = -1 + rad + pw;
			} else if (x < -1) {
				r = -1;
				reset();
			}
		} else if (x + rad + pw > 1) {
			if(onPaddle(o)){
				float i = (y - o) * 2 / pl;
				double a = Math.PI + Math.PI / 4 * i;
				float v = vdef + i * i * m * vdef;
				vx = v * (float) Math.cos(a);
				vy = v * (float) Math.sin(a);
				x = 1 - rad - pw;
			} else if (x > 1){
				r = 1;
				reset();
			}
		}

		notifyListeners();
		
		return r;
	}
	
	private boolean onPaddle(float paddle){
		return paddle - pl / 2 - rad < y &&  y < paddle + pl / 2 + rad;
	}
	
	private int selectOpponentAction() {
		if (y < o )
			return -1;
		else if (y > o)
			return 1;
		else
			return 0;
	}

	@Override
	public Tensor getObservation() {
		return factory.createTensor(new float[] { x, y, vx, vy, p, o }, 6);
	}

	@Override
	public void reset() {
		x = y = p = o = 0;

		double r = Math.random();
		r = (r < 0.5) ? 3 * Math.PI / 4 + r * Math.PI : -Math.PI / 4 + (r - 0.5) * Math.PI;

		vx = vdef * (float) Math.cos(r);
		vy = vdef * (float) Math.sin(r);
	}
	
	private void notifyListeners(){
		synchronized(listeners){
			listeners.stream().forEach(l -> l.update(x, y, vx, vy, p, o));
		}
	}

	@Reference
	void setTensorFactory(TensorFactory factory) {
		this.factory = factory;
	}
	
	@Reference(cardinality = ReferenceCardinality.MULTIPLE, policy = ReferencePolicy.DYNAMIC)
	void addPongListener(PongListener l){
		listeners.add(l);
	}
	
	void removePongListener(PongListener l){
		listeners.remove(l);
	}

	@Override
	public float getBounds() {
		return b;
	}

	@Override
	public float getSpeed() {
		return vdef;
	}

	@Override
	public float getPaddleLength() {
		return pl;
	}

	@Override
	public float getPaddleWidth() {
		return pw;
	}

	@Override
	public float getBallRadius() {
		return rad;
	}
	
}
