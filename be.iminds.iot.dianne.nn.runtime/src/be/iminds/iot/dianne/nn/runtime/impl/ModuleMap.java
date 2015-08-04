package be.iminds.iot.dianne.nn.runtime.impl;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

public class ModuleMap<T> {

	private HashMap<Key, T> map = new HashMap<>();
	private HashMap<UUID, Set<UUID>> keyMap = new HashMap<>();
	
	public synchronized T get(UUID moduleId, UUID nnId){
		return map.get(new Key(moduleId, nnId));
	}
	
	public synchronized void put(UUID moduleId, UUID nnId, T value){
		map.put(new Key(moduleId, nnId), value);
		Set<UUID> nns = keyMap.get(moduleId);
		if(nns==null){
			nns = new HashSet<>();
			keyMap.put(moduleId, nns);
		}
		nns.add(nnId);
	}
	
	public synchronized T remove(UUID moduleId, UUID nnId){
		Set<UUID> nns = keyMap.get(moduleId);
		nns.remove(nnId);
		if(nns.size()==0){
			keyMap.remove(moduleId);
		}
		return map.remove(new Key(moduleId, nnId));
	}
	
	public synchronized boolean containsKey(UUID moduleId, UUID nnId){
		return map.containsKey(new Key(moduleId, nnId));
	}
	
	public synchronized boolean containsKey(UUID moduleId){
		return keyMap.containsKey(moduleId);
	}
	
	public Collection<T> values(){
		return map.values();
	}
	
	private class Key {
		private final UUID moduleId;
	    private final UUID nnId;

	    public Key(UUID moduleId, UUID nnId) {
	        this.moduleId = moduleId;
	        this.nnId = nnId;
	    }

	    @Override
	    public boolean equals(Object o) {
	        if (this == o) return true;
	        if (!(o instanceof ModuleMap.Key)) return false;
	        Key key = (Key) o;
	        return moduleId.equals(key.moduleId) && nnId.equals(key.nnId);
	    }

	    @Override
	    public int hashCode() {
	        int result = moduleId.hashCode();
	        result = 31 * result + nnId.hashCode();
	        return result;
	    }
	}
	
	public class Entry<T> {
		public final UUID moduleId;
		public final UUID nnId;
		public final T value;
		
		public Entry(UUID moduleId, UUID nnId, T value){
			this.moduleId = moduleId;
			this.nnId = nnId;
			this.value = value;
		}
	}
	
	public Iterator<Entry<T>> iterator(){
		return new Iterator<ModuleMap<T>.Entry<T>>() {
			Iterator<Map.Entry<Key, T>> it = map.entrySet().iterator();
			
			@Override
			public boolean hasNext() {
				return it.hasNext();
			}

			@Override
			public ModuleMap<T>.Entry<T> next() {
				Map.Entry<Key, T> e = it.next();
				return new Entry<T>(e.getKey().moduleId, e.getKey().nnId, e.getValue());
			}
		};
	}
}
