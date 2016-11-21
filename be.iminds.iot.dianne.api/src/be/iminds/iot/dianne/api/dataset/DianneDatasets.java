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
package be.iminds.iot.dianne.api.dataset;

import java.util.List;
import java.util.Map;

/**
 * Interface to configure and manage all Datasets
 *  
 * @author tverbele
 *
 */
public interface DianneDatasets {

	/**
	 * @return list of all available Datasets
	 */
	List<DatasetDTO> getDatasets();

	/**
	 * Get the service of the dataset by name
	 * @return the Dataset service
	 */
	Dataset getDataset(String name);
	
	/**
	 * Configure a dataset with a number of configuration options... will create
	 * the necessary dataset adapters if required and return the resulting Dataset service
	 * @param name dataset to return
	 * @param config configurations of possible adapters that are required
	 * @return configured Dataset service
	 */
	Dataset configureDataset(String name, Map<String, String> config);
	
	/**
	 * Release a customly configured dataset that was acquired by configureDataset
	 * @param d dataset that is no longer used
	 */
	void releaseDataset(Dataset d);

	/**
	 * Checks whether a dataset has labels to be used for classification
	 * @param dataset
	 * @return whether the dataset has labels or not
	 */
	boolean isClassificationDataset(String dataset);
	
	/**
	 * Checks whether a dataset is an experience pool
	 * @param dataset
	 * @return whether the dataset is an experience pool or not
	 */
	boolean isExperiencePool(String dataset);
}
