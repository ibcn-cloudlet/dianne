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

import java.util.ArrayList;

/**
 * Helper configuration class to configure a sampling range of a Dataset
 * 
 * Set either startIndex/endIndex, a range or directly an indices array.
 * Call indices() method to generate the indices from the specified config parameters.
 * 
 * @author tverbele
 *
 */
public class SamplingConfig {

	/**
	 * Start index of the dataset to use
	 */
	public int startIndex = -1;
	
	/**
	 * End index of the dataset to use
	 */
	public int endIndex = -1;
	
	/**
	 * Range specifier formatted as a:b,c:d,...
	 */
	public String range;
	
	/**
	 * Select specific individual indices of the dataset
	 */
	public int[] indices;
	
	public int[] indices(Dataset d){
		if(indices==null){
			if(range != null){
				indices = parseRange(range);
			} else if(startIndex != -1 && endIndex != -1){
				if(startIndex == -1)
					startIndex = 0;
				if(endIndex == -1){
					endIndex = d.size();
				}
				int index = startIndex;
				indices = new int[endIndex-startIndex];
				for(int i=0;i<indices.length;i++){
					indices[i] = index++;
				}
			}
		}
		return indices;
	}
	
	private static int[] parseRange(String range){
		ArrayList<Integer> list = new ArrayList<>();
		String[] subranges = range.split(",");
		for(String subrange : subranges){
			String[] s = subrange.split(":");
			if(s.length==2){
				for(int i=Integer.parseInt(s[0]);i<Integer.parseInt(s[1]);i++){
					list.add(i);
				}
			} else {
				list.add(Integer.parseInt(s[0]));
			}
		}
		int[] array = new int[list.size()];
		for(int i=0;i<list.size();i++){
			array[i] = list.get(i);
		}
		return array;
	}
}
