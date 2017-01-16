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
package be.iminds.iot.dianne.nn.util;

import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.util.Map;
import java.util.UUID;

public class DianneConfigHandler {

	// TODO use Object Conversion spec implementation for this?!
	@SuppressWarnings({ "unchecked", "rawtypes" })
	public static <T> T getConfig(Map<String, String> config, Class<T> c){
		T instance = null;
		try {
			instance = c.newInstance();
			
			for(Field f : c.getFields()){
				String value = config.get(f.getName());
				if(value != null){
					if(f.getType().isPrimitive()){
						if(f.getType().equals(Integer.TYPE)){
							f.setInt(instance, Integer.parseInt(value));
						} else if(f.getType().equals(Long.TYPE)){
							f.setLong(instance, Long.parseLong(value));
						} else if(f.getType().equals(Float.TYPE)){
							f.setFloat(instance, Float.parseFloat(value));
						} else if(f.getType().equals(Double.TYPE)){
							f.setDouble(instance, Double.parseDouble(value));
						} else if(f.getType().equals(Byte.TYPE)){
							f.setByte(instance, Byte.parseByte(value));
						} else if(f.getType().equals(Short.TYPE)){
							f.setShort(instance, Short.parseShort(value));
						} else if(f.getType().equals(Boolean.TYPE)){
							f.setBoolean(instance, Boolean.parseBoolean(value));
						}
					} else if(f.getType().isEnum()){
						f.set(instance, Enum.valueOf((Class<Enum>) f.getType(), value.toUpperCase()));
					} else if(f.getType().isArray()){
						String[] array = value.split(",");
						if(f.getType().getComponentType().equals(Integer.TYPE)){
							int[] intarray = new int[array.length];
							for(int i=0;i<intarray.length;i++){
								intarray[i] = Integer.parseInt(array[i]);
							}
							f.set(instance, intarray);
						} else if(f.getType().getComponentType().equals(Double.TYPE)){
							double[] doublearray = new double[array.length];
							for(int i=0;i<doublearray.length;i++){
								doublearray[i] = Double.parseDouble(array[i]);
							}
							f.set(instance, doublearray);
						} else if(f.getType().getComponentType().equals(Float.TYPE)){
							float[] floatarray = new float[array.length];
							for(int i=0;i<floatarray.length;i++){
								floatarray[i] = Float.parseFloat(array[i]);
							}
							f.set(instance, floatarray);
						} else if(f.getType().getComponentType().equals(UUID.class)){
							UUID[] uuidarray = new UUID[array.length];
							for(int i=0;i<uuidarray.length;i++){
								uuidarray[i] = UUID.fromString(array[i]);
							}
							f.set(instance, uuidarray);
						} else {
							f.set(instance, array);
						}
					} else {
						f.set(instance, value);
					}
				}
			}
		
			printConfig(instance);
		} catch(Exception e){
			e.printStackTrace();
		}
		return instance;
	}
	
	private static void printConfig(Object config){
		String name = config.getClass().getName();
		name = name.substring(name.lastIndexOf(".")+1);
		name = name.substring(0, name.length()-6);
		System.out.println(name);
		System.out.println("---");
		for(Field f : config.getClass().getFields()){
			try {
				if(f.getType().isArray()){
					String s = "* "+f.getName()+" = [";
					Object array = f.get(config);
					int l = Array.getLength(array);
					for(int i=0;i<l;i++){
						s+= Array.get(array, i);
						if(i != l-1){
							s+= ", ";
						}
					}
					s+="]";
					System.out.println(s);
				} else {
					String s = ""+f.get(config);
					if(s.length() > 100){
						s = s.substring(0, 100)+"...";
					}
					System.out.println("* "+f.getName()+" = "+s);
				}
			} catch (IllegalArgumentException e) {
			} catch (IllegalAccessException e) {
			}
		}
		System.out.println("---");
	}
}
