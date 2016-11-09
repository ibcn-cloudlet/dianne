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
package be.iminds.iot.dianne.api.coordinator;

public class Status {

	public final int queued;
	public final int running;
	
	public final int learn;
	public final int eval;
	public final int act;
	public final int idle;
	public final int devices;

	public final long spaceLeft;
	
	public final long bootTime;
	
	public Status(int queued, int running,
			int learn, int eval, int act, int idle, int devices,
			long spaceLeft, long bootTime){
		this.queued = queued;
		this.running = running;
		this.learn = learn;
		this.eval = eval;
		this.act = act;
		this.idle = idle;
		this.devices = devices;
		this.spaceLeft = spaceLeft;
		this.bootTime = bootTime;
	}
	
}
