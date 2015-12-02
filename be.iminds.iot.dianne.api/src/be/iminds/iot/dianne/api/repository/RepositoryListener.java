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
package be.iminds.iot.dianne.api.repository;

import java.util.Collection;
import java.util.UUID;

/**
 * Listener to get notifications when the Repository has updates for a given moduleId.
 * 
 * Listeners are registered with a String[] targets service property, indicating a number of
 * moduleId:tag1:tag2 strings that identify the moduleId/tags this listener is interested in.
 * 
 * One of the tags could also be used to match a neural network instance id one is interested in.
 * For example:
 * targets={"uuid1:run",":uuid2:test"}
 * listens to updates with "run" tag of module with uuid uuid1 and to updates of neural network
 * instance with id uuid2 and tagged with "test"
 * 
 * @author tverbele
 *
 */
public interface RepositoryListener {

	/**
	 * Notify listener of a parameter update for a collection of moduleIds and optional tags 
	 * 
	 * @param nnId the nn instance these parameters originate from
	 * @param moduleIds moduleIds whos parameters were updated
	 * @param tag optional tags
	 */
	public void onParametersUpdate(UUID nnId, Collection<UUID> moduleIds, String... tag);
	
}
