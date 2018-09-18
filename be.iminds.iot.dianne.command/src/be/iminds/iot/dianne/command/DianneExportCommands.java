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

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

import org.apache.felix.service.command.Descriptor;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.util.DianneExporter;

/**
 * Separate component for export commands
 */
@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=jar"},
		immediate=true)
public class DianneExportCommands {

	private DianneExporter exporter;
	
	@Descriptor("Export a neural network to a .jar.")
	public void jar(String nnName, String... properties) throws IOException {
		String fileName = nnName+".jar";
		
		String[] tags = null;
		Map<String, String> config = ConfigurationParser.parse(properties);
		if(config.containsKey("tag")) {
			tags = config.get("tag").split(",");
		}
		
		File f = new File(fileName);
		Path path = Paths.get(f.getAbsolutePath());
		Files.write(path, exporter.export(nnName, config, tags));
	}
	
	@Reference
	void setDianneExporter(DianneExporter e){
		this.exporter = e;
	}
	
}
