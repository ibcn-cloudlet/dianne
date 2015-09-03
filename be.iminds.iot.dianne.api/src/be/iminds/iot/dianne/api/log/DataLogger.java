package be.iminds.iot.dianne.api.log;

/**
 * The DataLogger provides an interface for log various values and will print them
 * out nicely and/or calculate a running average.
 * 
 * @author tverbele
 *
 */
public interface DataLogger {

	void log(String label, String[] keys, float... values);
	
}
