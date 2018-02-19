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
package be.iminds.iot.dianne.coordinator;

import java.util.UUID;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.aiolos.info.NodeInfo;
import be.iminds.aiolos.platform.api.PlatformManager;
import be.iminds.iot.dianne.api.coordinator.Device;

/**
 * Use AIOLOS Platform manager to query device info ...
 * 
 * Might be replaced later e.g. using ClusterInfo spec
 * 
 * This is kinda hacky cause we rely on the name (AilosDeviceManagerImpl) being alphabetically before
 * DianneCoordinatorImpl in order to make things work ...
 * 
 * @author tverbele
 *
 */
@Component
public class AiolosDeviceManagerImpl implements DeviceManager {

	private PlatformManager aiolos;

	@Override
	public Device getDevice(UUID id) {
		NodeInfo n = aiolos.getNode(id.toString());
		String name = n.getName();
		String arch = n.getArch();
		String os = n.getOS();
		String ip = n.getIP();
		Device d = new Device(id, name, arch, os, ip);
		return d;
	}

	@Reference
	void setPlatformManager(PlatformManager pm) {
		this.aiolos = pm;
	}
}
