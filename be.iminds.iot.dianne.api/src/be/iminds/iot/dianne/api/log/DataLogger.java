package be.iminds.iot.dianne.api.log;

/**
 * The DataLogger provides an interface for log various values and will print them
 * out nicely and/or calculate a running average.
 * 
 * @author tverbele
 *
 */
public interface DataLogger {

	/**
	 * Log a new entry - keys.length should equals number of values!
	 * @param label label for this set of key-values
	 * @param keys key labels for the values
	 * @param values the values to log
	 */
	void log(String label, String[] keys, float... values);

	/**
	 * Set an other than default interval for log messages with this label
	 * @param label 
	 * @param interval
	 */
	void setInterval(String label, int interval);
	
	/**
	 * Set an other than default alpha value for the running average of values with this key
	 * @param key
	 * @param alpha
	 */
	void setAlpha(String key, float alpha);
}
