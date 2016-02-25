package be.iminds.iot.dianne.coordinator;

import java.util.concurrent.LinkedBlockingQueue;

public class CircularBlockingQueue<T> extends LinkedBlockingQueue<T> {

	private int limit;
	
	public CircularBlockingQueue(int limit){
		super();
		this.limit = limit;
	}
	
    @Override
    public boolean add(T item) {
        super.add(item);
        while (size() > limit) { super.remove(); }
        return true;
    }
	
}
