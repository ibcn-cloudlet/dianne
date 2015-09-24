package be.iminds.iot.dianne.rl.worker;

import java.util.HashMap;
import java.util.Map;

import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.rl.Agent;
import be.iminds.iot.dianne.api.rl.ExperiencePool;

/**
 * Worker component that bootstraps a node with an Agent, Learner and local ExperiencePool
 * that starts generating experience and learning on this experience.
 *  
 * @author tverbele
 *
 */
@Component(immediate=true, configurationPolicy=ConfigurationPolicy.REQUIRE)
public class RLWorker {
	
	private Agent agent;
	private Learner learner;
	private Map<String, ExperiencePool> pools = new HashMap<String, ExperiencePool>();

	
	@Activate
	public void activate(Map<String, Object> properties){
		final Map<String, String> config = new HashMap<String, String>();
		properties.entrySet().stream().forEach(e -> config.put(e.getKey(), e.getValue().toString()));
		if(!config.containsKey("tag")){
			config.put("tag", "run");
		}
		
		final String pool = (String)properties.get("pool");
		final String nn = (String)properties.get("nn");
		final String env = (String)properties.get("environment");
		
		Thread t = new Thread( () -> {
			// wait until experience pool configured
			synchronized(pools){
				if(pool!=null && !pools.containsKey(pool)){
					try {
						pools.wait(20000);
					} catch (Exception e1) {
					}
				}
			}
			
			try {
				agent.act(nn, env, pool, config);
			} catch(Exception e){
				System.err.println("Failed to start agent");
				e.printStackTrace();
			}
			try {
				learner.learn(nn, pool, config);
			} catch(Exception e){
				System.err.println("Failed to start learner");
				e.printStackTrace();
			}
		});
		t.start();
	}
	
	@Reference
	void setAgent(Agent agent){
		this.agent = agent;
	}
	
	@Reference
	void setLearner(Learner l){
		this.learner = l;
	}
	
	@Reference(cardinality = ReferenceCardinality.MULTIPLE, policy = ReferencePolicy.DYNAMIC)
	void addExperiencePool(ExperiencePool pool, Map<String, Object> properties) {
		String name = (String) properties.get("name");
		synchronized(pools){
			this.pools.put(name, pool);
			this.pools.notifyAll();
		}
	}

	void removeExperiencePool(ExperiencePool pool, Map<String, Object> properties) {
		String name = (String) properties.get("name");
		synchronized(pools){
			this.pools.remove(name);
		}
	}
	
}
