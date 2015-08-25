package be.iminds.iot.dianne.rl.agent;

import java.util.HashMap;
import java.util.Map;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.rl.Agent;
import be.iminds.iot.dianne.api.rl.Environment;
import be.iminds.iot.dianne.api.rl.ExperiencePool;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component
public class GreedyDeepQAgent implements Agent {

	private TensorFactory factory;
	private Map<String, ExperiencePool> pools = new HashMap<String, ExperiencePool>();
	private Map<String, Environment> envs = new HashMap<String, Environment>();

	private Thread thread;
	private volatile boolean running;

	private float epsilon = 0.1f;

	@Reference
	void setTensorFactory(TensorFactory factory) {
		this.factory = factory;
	}

	@Reference(cardinality = ReferenceCardinality.MULTIPLE, policy = ReferencePolicy.DYNAMIC)
	public void addExperiencePool(ExperiencePool pool, Map<String, Object> properties) {
		String name = (String) properties.get("name");
		this.pools.put(name, pool);
	}

	public void removeExperiencePool(ExperiencePool pool, Map<String, Object> properties) {
		String name = (String) properties.get("name");
		this.pools.remove(name);
	}

	@Reference(cardinality = ReferenceCardinality.MULTIPLE, policy = ReferencePolicy.DYNAMIC)
	public void addEnvironment(Environment env, Map<String, Object> properties) {
		String name = (String) properties.get("name");
		this.envs.put(name, env);
	}

	public void removeEnvironment(Environment env, Map<String, Object> properties) {
		String name = (String) properties.get("name");
		this.envs.remove(name);
	}

	@Activate
	void activate(BundleContext context) {
		String epsilon = context.getProperty("be.iminds.iot.dianne.rl.agent.greedy");
		if (epsilon != null)
			this.epsilon = Float.parseFloat(epsilon);
	}

	@Deactivate
	void deactivate() {
		stop();
	}

	@Override
	public synchronized void act(String nnName, String environment, String experiencePool, Map<String, String> config)
			throws Exception {
		if (running)
			throw new Exception("Already running an Agent here");
		else if (environment == null || !envs.containsKey(environment))
			throw new Exception("Environment " + environment + " is null or not available");
		else if (experiencePool != null && !pools.containsKey(experiencePool))
			throw new Exception("ExperiencePool " + experiencePool + "is not available");

		float epsilon = this.epsilon;
		if (config.containsKey("epsilon"))
			epsilon = Float.parseFloat(config.get("epsilon"));

		Environment env = envs.get(environment);
		ExperiencePool pool = pools.get(experiencePool);

		thread = new Thread(new GreedyDeepQRunnable(env, pool, epsilon));
		running = true;
		thread.start();
	}

	@Override
	public synchronized void stop() {
		try {
			if (thread != null && thread.isAlive()) {
				running = false;
				thread.join();
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

	private class GreedyDeepQRunnable implements Runnable {

		private final Environment env;
		private final ExperiencePool pool;
		private final float epsilon;

		public GreedyDeepQRunnable(Environment env, ExperiencePool pool, float epsilon) {
			this.env = env;
			this.pool = pool;
			this.epsilon = epsilon;
		}

		@Override
		public void run() {
			Tensor state = env.getObservation();

			while (running) {
				//TODO q = nn.forward(state)
				Tensor q = factory.createTensor(3);
				q.rand();

				Tensor action = factory.createTensor(q.size());
				action.fill(-1);

				if (Math.random() < epsilon) {
					action.set(1, (int) (Math.random() * action.size()));
				} else {
					action.set(1, factory.getTensorMath().argmax(q));
				}

				float reward = env.performAction(action);
				Tensor nextState = env.getObservation();

				if (pool != null)
					pool.addSample(state, action, reward, nextState);

				state = nextState;
			}
		}

	}

}
