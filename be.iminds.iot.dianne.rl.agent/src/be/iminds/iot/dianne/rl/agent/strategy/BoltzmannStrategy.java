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
package be.iminds.iot.dianne.rl.agent.strategy;

import java.util.Map;

import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.agent.strategy.config.BoltzmannConfig;
import be.iminds.iot.dianne.tensor.ModuleOps;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

@Component(property={"strategy=BOLTZMANN",
			"aiolos.proxy=false"})
public class BoltzmannStrategy implements ActionStrategy {
	
	private BoltzmannConfig config;
	
	private String[] loglabels = new String[]{"Q0", "Q1", "Q2", "temperature"};
	
	@Override
	public Tensor selectActionFromOutput(Tensor output, long i) {
		
		Tensor action = new Tensor(output.size());
		action.fill(-1);
		
		double temperature = config.temperatureMin + (config.temperatureMax - config.temperatureMin) * Math.exp(-i * config.temperatureDecay);
		
		TensorOps.div(output, output, (float) temperature);
		ModuleOps.softmax(output, output);
		
		double s = 0, r = Math.random();
		int a = 0;
		
		while((s += output.get(a)) < r)
			a++;
		
		action.set(1, a);
		
		return action;
	}

	@Override
	public void configure(Map<String, String> config) {
		this.config = DianneConfigHandler.getConfig(config, BoltzmannConfig.class);
	}

}
