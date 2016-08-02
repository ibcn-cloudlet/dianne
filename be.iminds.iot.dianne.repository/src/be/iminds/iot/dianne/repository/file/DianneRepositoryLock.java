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
package be.iminds.iot.dianne.repository.file;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

public class DianneRepositoryLock {

	private final Map<UUID, Set<Thread>> reading = new HashMap<>();
	private final Map<UUID, Thread> writing = new HashMap<>();
	private final Map<Thread, Integer> reentrantCount = new HashMap<>(); 
	
	public void read(UUID moduleId) throws InterruptedException {
		synchronized(this){
			// check if someone (not us) is currently writing
			boolean assign = false;
			while(!assign){
				Thread t = writing.get(moduleId);
				// assign if no-one owns write lock (besides us) 
				if( t == null || t == Thread.currentThread()){
					assign = true;
				}
				
				if(!assign){
					this.wait();
				}
			}
			
			// we made it - we can read!
			// mark our thread as reading
			Set<Thread> readingThreads = reading.get(moduleId);
			if(readingThreads == null){
				readingThreads = new HashSet<>();
				reading.put(moduleId, readingThreads);
			}
			readingThreads.add(Thread.currentThread());
			
			// update counter
			Integer count = reentrantCount.get(Thread.currentThread());
			if(count == null){
				reentrantCount.put(Thread.currentThread(), 1);
			} else {
				reentrantCount.put(Thread.currentThread(), count+1);
			}
		}
	}
	
	public void write(UUID moduleId) throws InterruptedException{
		synchronized(this){
			// check if some else is currently reading or writing
			boolean assign = false;
			while(!assign){
				Thread t = writing.get(moduleId);
				Set<Thread> r = reading.get(moduleId);
				
				// assign if nobody else is writing
				if(t == null || t == Thread.currentThread()){
					assign = true;
				}
				
				// but also nobody is currently reading (besides us)
				if(r != null){
					if(r.size() > 1 || !r.contains(Thread.currentThread())){
						assign = false;
					}
				}
				
				if(!assign){
					this.wait();
				}
			}
			
			// we made it - we can write!
			writing.put(moduleId, Thread.currentThread());
			
			// update counter
			Integer count = reentrantCount.get(Thread.currentThread());
			if(count == null){
				reentrantCount.put(Thread.currentThread(), 1);
			} else {
				reentrantCount.put(Thread.currentThread(), count+1);
			}
		}
	}
	
	public void free(UUID moduleId){
		synchronized(this){
			// check count
			Integer count = reentrantCount.get(Thread.currentThread());
			if(count == null){
				throw new RuntimeException("Cannot free the lock for module "+moduleId+" from this thread");
			}
			if((int)count != 1){
				reentrantCount.put(Thread.currentThread(), count-1);
				return;
			} else {
				reentrantCount.remove(Thread.currentThread());
			}
			
			// release any read/write lock
			if(writing.get(moduleId) == Thread.currentThread()){
				writing.remove(moduleId);
			}
			Set<Thread> r = reading.get(moduleId);
			if(r != null){
				r.remove(Thread.currentThread());
				if(r.size() == 0){
					reading.remove(moduleId);
				}
			}
			
			this.notifyAll();
		}
	}
}
