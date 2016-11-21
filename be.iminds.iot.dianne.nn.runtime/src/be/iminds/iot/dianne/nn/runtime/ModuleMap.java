/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  iMinds - IBCN - UGent
 *
 * This file is part of DIANNE.
 *
 * DIANNE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.nn.runtime;

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
	        @SuppressWarnings("unchecked")
			Key key = (Key) o;
	        return moduleId.equals(key.moduleId) && nnId.equals(key.nnId);
	    }

	    @Override
	    public int hashCode() {
	        int result = moduleId.hashCode();
	        result = 31 * result + nnId.hashCode();
	        return result;
	    }
	    
	    public String toString(){
	    	return nnId.toString()+"-"+moduleId.toString();
	    }
	}
	
	public class Entry<S> {
		public final UUID moduleId;
		public final UUID nnId;
		public final S value;
		
		public Entry(UUID moduleId, UUID nnId, S value){
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
