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

import java.util.UUID;

public class Device {

	public final UUID id;
	public final String name;
	public final String arch;
	public final String os;
	
	public String ip;
	
	public boolean learn = false;
	public boolean eval = false;
	public boolean act = false;
	
	public Device(UUID id, String name, String arch, String os, String ip){
		this.id = id;
		this.name = name;
		this.arch = arch;
		this.os = os;
		this.ip = ip;
	}
	
}
