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
 *     Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.nn.util.random;

public class Process {
	
	private final Distribution interArrival;
	
	private long start;
	private long current;
	
	public Process(Distribution interArrival){
		this(0l,interArrival);
	}
	
	public Process(long start, Distribution interArrival){
		this.start = start;
		this.current = this.start;
		this.interArrival = interArrival;
	}
	
	public Process(Process process){
		this(process.start,process.interArrival);
	}
	
	public long nextEvent(){
		return this.current += interArrival.nextLong();
	}
	
	public void reset(){
		reset(start);
	}
	
	public void reset(long start){
		this.start = start;
		this.current = this.start;
	}
}
