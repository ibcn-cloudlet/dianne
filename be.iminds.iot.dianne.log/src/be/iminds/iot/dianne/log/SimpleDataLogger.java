package be.iminds.iot.dianne.log;

import java.util.HashMap;
import java.util.Map;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.api.log.DataLogger;

@Component
public class SimpleDataLogger implements DataLogger {

	private float defaultAlpha = 0.01f;
	private int defaultInterval = 1000;
	
	private Map<String, Float> averages = new HashMap<>();
	private Map<String, Integer> counts = new HashMap<>();
	
	private Map<String, Integer> intervals = new HashMap<>();
	private Map<String, Float> alphas = new HashMap<>();
	
	
	@Activate
	public void activate(BundleContext context){
		String a = context.getProperty("be.iminds.iot.dianne.log.alpha");
		if (a != null)
			this.defaultAlpha = Float.parseFloat(a);

		String i = context.getProperty("be.iminds.iot.dianne.log.interval");
		if (i != null)
			this.defaultInterval = Integer.parseInt(i);
	}
	
	@Override
	public void log(String label, String[] keys, float... values) {
		assert keys.length==values.length;
		for(int i=0;i<keys.length;i++){
			Float old = averages.get(keys[i]);
			if(old==null){
				averages.put(keys[i], values[i]);
			} else {
				Float alpha = alphas.get(keys[i]);
				if(alpha==null){
					alpha = defaultAlpha;
				}
				
				averages.put(keys[i], (1-alpha) * old  + alpha * values[i]);
			}
		}
		
		Integer interval = intervals.get(label);
		if(interval==null){
			interval = defaultInterval;
		}
		
		if(interval.intValue() < 1){
			// print values directly
			StringBuilder b = new StringBuilder();
			b.append("[").append(label).append("]\t");
			for(int i=0;i<keys.length;i++){
				b.append(keys[i]).append("\t").append(values[i]).append("\t");
			}
			System.out.println(b.toString());

		} else {
			Integer count = counts.get(label);
			if(count==null){
				count = 0;
			} 
			count = count + 1;
			
			if(count.intValue() == interval.intValue()){
				// print averages and reset count
				StringBuilder b = new StringBuilder();
				b.append("[").append(label).append("]\t");
				for(int i=0;i<keys.length;i++){
					b.append(keys[i]).append("\t").append(averages.get(keys[i])).append("\t");
				}
				System.out.println(b.toString());
				
				count = 0;
			}
			counts.put(label, count);
		}
	}

	@Override
	public void setInterval(String label, int interval) {
		intervals.put(label, interval);
	}

	@Override
	public void setAlpha(String key, float alpha) {
		alphas.put(key, alpha);
	}
}
