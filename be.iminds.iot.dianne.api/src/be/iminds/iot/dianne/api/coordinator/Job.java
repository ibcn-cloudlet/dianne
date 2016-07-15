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

import java.util.List;
import java.util.Map;
import java.util.UUID;

public class Job {

	public enum Type {
		LEARN,
		EVALUATE,
		ACT
	}
	
	public interface Category {}
	
	public enum LearnCategory implements Category {
		FF,
		RNN,
		RL
	}
	
	public enum EvaluationCategory implements Category {
		CLASSIFICATION,
		CRITERION
	}
	
	public final UUID id;
	public final String name;
	public final Type type;
	public final Category category;
	public final String[] nn;
	public final String dataset;
	public final Map<String, String> config;
	
	public Job(UUID id, String name, Type type, Category category, String d, Map<String, String> c,  String... nn){
		this.id = id;
		this.name = name;
		this.type = type;
		this.category = category;
		this.nn = nn;
		this.dataset = d;
		this.config = c;
	}
	
	public long submitted;
	public long started;
	public long stopped;
	public List<UUID> targets;
	
}
