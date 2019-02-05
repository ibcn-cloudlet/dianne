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
package be.iminds.iot.dianne.onnx;

import java.util.Map;
import java.util.UUID;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.onnx.api.OnnxConverter;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(immediate=true, 
	property={"osgi.command.scope=dianne",
		  "osgi.command.function=fromOnnx",
		  "osgi.command.function=toOnnx",
		  "osgi.command.function=loadOnnx"
		  
	})
public class DianneOnnx implements OnnxConverter {

	private DianneRepository repository;
	
	@Reference
	void setDianneRepository(DianneRepository r) {
		this.repository = r;
	}
	
	public NeuralNetworkDTO fromOnnx(String onnxFile) {
		OnnxImporter importer = new OnnxImporter(onnxFile);
		NeuralNetworkDTO nn = importer.getNN();
		repository.storeNeuralNetwork(nn);
		repository.storeParameters(nn.name, importer.getParameters());
		return nn;
	}
	
	public NeuralNetworkDTO fromOnnx(String onnxFile, String name) {
		OnnxImporter importer = new OnnxImporter(onnxFile);
		NeuralNetworkDTO nn = importer.getNN(name);
		repository.storeNeuralNetwork(nn);
		repository.storeParameters(nn.name, importer.getParameters());
		return nn;
	}
	
	
	public void toOnnx(String onnxFile, String nnName, String... tag) {
		try {
			NeuralNetworkDTO nn = repository.loadNeuralNetwork(nnName);
			Map<UUID, Tensor> parameters = repository.loadParameters(nnName, tag);
			OnnxExporter exporter = new OnnxExporter(nn, parameters);
			exporter.export(onnxFile);
		} catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Failed to export "+nnName, e);
		}
	}
	
	public void loadOnnx(String onnxFile, String nnName, String... tags) {
		try {
			NeuralNetworkDTO nn = repository.loadNeuralNetwork(nnName);
			OnnxLoader loader = new OnnxLoader(onnxFile, nn);
			repository.storeParameters(nn.name, loader.getParameters(), tags);
		} catch (Exception e) {
			throw new RuntimeException("Failed to load "+nnName, e);
		}
	}
}