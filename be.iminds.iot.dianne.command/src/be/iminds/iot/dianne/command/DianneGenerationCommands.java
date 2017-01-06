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
package be.iminds.iot.dianne.command;

import org.apache.felix.service.command.Descriptor;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

@Component(service = Object.class, 
	property = { 
		"osgi.command.scope=dianne",
		"osgi.command.function=generate" }, 
	immediate = true)
public class DianneGenerationCommands {

	private Dianne dianne;
	private DiannePlatform platform;

	@Descriptor("Generate a string sequence with a neural network")
	public void generate(
			@Descriptor("neural network to use for generation")
			String nnName, 
			@Descriptor("start string to feed to the neural net first")
			String start, 
			@Descriptor("length of the string to generate")
			int n, 
			@Descriptor("optional tags of the neural net to load")
			String... tags) {
		// forward of a rnn
		NeuralNetworkInstanceDTO nni = null;
		try {
			nni = platform.deployNeuralNetwork(nnName, "test rnn", tags);
			NeuralNetwork nn = dianne.getNeuralNetwork(nni).getValue();

			System.out.print(start);

			for (int i = 0; i < start.length() - 1; i++) {
				nextChar(nn, start.charAt(i));
			}

			char c = start.charAt(start.length() - 1);
			for (int i = 0; i < n; i++) {
				c = nextChar(nn, c);
				System.out.print(""+c);
			}

		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			platform.undeployNeuralNetwork(nni);
		}
	}

	private char nextChar(NeuralNetwork nn, char current) {
		// construct input tensor
		String[] labels = nn.getOutputLabels();
		if (labels == null) {
			throw new RuntimeException(
					"Neural network " + nn.getNeuralNetworkInstance().name + " is not trained and has no labels");
		}
		Tensor in = new Tensor(labels.length);
		in.fill(0.0f);
		int index = 0;
		for (int i = 0; i < labels.length; i++) {
			if (labels[i].charAt(0) == current) {
				index = i;
				break;
			}
		}
		in.set(1.0f, index);

		// forward
		Tensor out = nn.forward(in);

		// select next, sampling from (Log)Softmax output
		if (TensorOps.min(out) < 0) {
			// assume logsoftmax output, take exp
			out = TensorOps.exp(out, out);
		}

		double s = 0, r = Math.random();
		int o = 0;
		while (o < out.size() && (s += out.get(o)) < r) {
			o++;
		}

		return labels[o].charAt(0);
	}

	@Reference
	void setDianne(Dianne d) {
		this.dianne = d;
	}

	@Reference
	void setDiannePlatform(DiannePlatform p) {
		this.platform = p;
	}

}
