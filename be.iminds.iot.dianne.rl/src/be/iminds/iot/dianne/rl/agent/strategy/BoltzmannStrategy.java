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
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;

import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.tensor.ModuleOps;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

@Component(property={"strategy=boltzmann",
			"aiolos.proxy=false"})
public class BoltzmannStrategy implements ActionStrategy {
	
	private double temperatureMax = 1e0;
	private double temperatureMin = 1e0;
	private double temperatureDecay = 1e-6;
	
	private DataLogger logger = null;
	private String[] loglabels = new String[]{"Q0", "Q1", "Q2", "temperature"};
	
	@Reference(cardinality = ReferenceCardinality.OPTIONAL)
	void setDataLogger(DataLogger l){
		this.logger = l;
		this.logger.setAlpha("temperature", 1f);
		this.logger.setAlpha("Q0", 1f);
		this.logger.setAlpha("Q1", 1f);
		this.logger.setAlpha("Q2", 1f);
	}
	
	@Override
	public Tensor selectActionFromOutput(Tensor output, long i) {
		
		Tensor action = new Tensor(output.size());
		action.fill(-1);
		
		double temperature = temperatureMin + (temperatureMax - temperatureMin) * Math.exp(-i * temperatureDecay);
		
		if(logger!=null){
			logger.log("AGENT", loglabels, output.get(0), output.get(1), output.get(2), (float) temperature);
		}
		
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
		if (config.containsKey("temperatureMax"))
			temperatureMax = Double.parseDouble(config.get("temperatureMax"));
		
		if (config.containsKey("temperatureMin"))
			temperatureMin = Double.parseDouble(config.get("temperatureMin"));
		
		if (config.containsKey("temperatureDecay"))
			temperatureDecay = Double.parseDouble(config.get("temperatureDecay"));
		
		System.out.println("Boltzmann Action Selection");
		System.out.println("* temperature max = "+temperatureMax);
		System.out.println("* temperature min = "+temperatureMin);
		System.out.println("* temperature decay = "+temperatureDecay);
		System.out.println("---");
	}

}
