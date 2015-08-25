package be.iminds.iot.dianne.rl.pong;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.rl.Environment;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

/**
 * Simple Pong environment in which an agents plays against an AI trying to
 * match the vertical position of the ball.
 * 
 * @author smbohez
 *
 */
@Component(immediate = true, property = { "name=Pong", "aiolos.callback=be.iminds.iot.dianne.api.rl.Environment" })
public class PongEnvironment implements Environment {

	private TensorFactory factory;

	private float l = 0.4f, vdef = 0.04f;

	private float x, y, vx, vy, p, o;

	@Reference
	void setTensorFactory(TensorFactory factory) {
		this.factory = factory;
	}

	@Activate
	void activate(BundleContext context) {
		String l = context.getProperty("be.iminds.iot.dianne.rl.pong.paddlelength");
		if (l != null)
			this.l = Float.parseFloat(l);

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
		float d_o = vdef * selectOponentAction();

		p += d_p;
		o += d_o;

		p = Math.min(1 - l / 2, p);
		p = Math.max(l / 2 - 1, p);

		o = Math.min(1 - l / 2, o);
		o = Math.max(l / 2 - 1, o);

		x += vx;
		y += vy;

		if (y < -1) {
			y = -2 - y;
			vy = -vy;
		} else if (y > 1) {
			y = 2 - y;
			vy = -vy;
		}

		float r = 0;

		if (x < -1) {
			float i = vy / vx * (-1 - x) + y;

			if (p - l / 2 < i && i < p + l / 2) {
				float d = (float) Math.sqrt(((x + 1) * (x + 1) + (y - i) * (y - i)) / (vx * vx + vy * vy));
				i = (i - p) * 2 / l;
				double a = Math.PI / 4 * i;
				float v = vdef + i * i * 7 * vdef;
				vx = v * (float) Math.cos(a);
				vy = v * (float) Math.sin(a);
				x = -1 + d * vx;
				y = i + d * vy;
			} else {
				r = -1;
				reset();
			}
		} else if (x > 1) {
			float i = vy / vx * (1 - x) + y;

			if (o - l / 2 < i && i < o + l / 2) {
				float d = (float) Math.sqrt(((x - 1) * (x - 1) + (y - i) * (y - i)) / (vx * vx + vy * vy));
				i = (i - p) * 2 / l;
				double a = Math.PI + Math.PI / 4 * i;
				float v = vdef + i * i * 7 * vdef;
				vx = v * (float) Math.cos(a);
				vy = v * (float) Math.sin(a);
				x = 1 + d * vx;
				y = i + d * vy;
			} else {
				r = 1;
				reset();
			}
		}

		return r;
	}

	private float selectOponentAction() {
		if (y < o - l / 2)
			return -1;
		else if (y > o + l / 2)
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

}
