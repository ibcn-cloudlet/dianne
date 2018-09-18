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
 *     Tim Verbelen
 *******************************************************************************/
package be.iminds.iot.dianne.nn.util.export;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.UUID;
import java.util.jar.Attributes;
import java.util.jar.Manifest;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import org.osgi.framework.BundleContext;
import org.osgi.framework.Constants;
import org.osgi.framework.Version;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.util.DianneExporter;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.nn.util.DianneJSONConverter;
import be.iminds.iot.dianne.tensor.Tensor;

@Component
public class DianneExporterImpl implements DianneExporter {

	private BundleContext context;
	private DianneRepository repository;

	@Activate
	void activate(BundleContext c) {
		this.context = c;
	}

	@Reference
	void setRepository(DianneRepository r) {
		this.repository = r;
	}

	@Override
	public byte[] export(String name, String... tags) throws IOException {
		return export(name, null, tags);
	}
	
	@Override
	public byte[] export(String name, Map<String, String> properties, String... tags) throws IOException {
		try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
				ZipOutputStream zos = new ZipOutputStream(baos);
				DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(zos))) {
			String version = null;
			// check whether we have an actual version property
			if(properties.containsKey("version")) {
				String v = properties.get("version");
				if(isVersion(v)) {
					version = v;
				}
			}
			// maybe there is a tag that can be used as version?
			if(version == null) {
				for (String t : tags) {
					if (isVersion(t)) {
						version = t;
						break;
					}
				}
			}
			// if no version specified, use 1.0.0
			if(version == null) {
				version = "1.0.0";
			}

			// add manifest
			Manifest manifest = new Manifest();
			Attributes atts = manifest.getMainAttributes();
			atts.put(Attributes.Name.MANIFEST_VERSION, "1.0");
			atts.putValue(Constants.BUNDLE_MANIFESTVERSION, "2");
			atts.putValue(Constants.BUNDLE_NAME, "Dianne NN " + name);
			atts.putValue(Constants.BUNDLE_SYMBOLICNAME, "be.iminds.iot.dianne.nn." + name);
			atts.putValue(Constants.BUNDLE_VERSION, version);
			atts.putValue("NeuralNetwork", name);
			// TODO add requirement on a DIANNE runtime capability instead of Import-Package?
			atts.putValue("Import-Package", "be.iminds.iot.dianne.api.nn.runtime;version=\"" + dianneVersion() + "\"");
			
			// TODO do we need to add (some of) the properties as capabilities?
			// TODO do we need to specify a namespace for this?
			
			zos.putNextEntry(new ZipEntry("META-INF/MANIFEST.MF"));
			manifest.write(zos);
			zos.closeEntry();

			// add nn description
			NeuralNetworkDTO nn = repository.loadNeuralNetwork(name);
			// override properties if specified
			if(properties != null) {
				for(Entry<String, String> e : properties.entrySet()) {
					nn.properties.put(e.getKey(), e.getValue());
				}
			}
			String nnString = DianneJSONConverter.toJsonString(nn, true);

			zos.putNextEntry(new ZipEntry("modules.txt"));
			zos.write(nnString.getBytes());
			zos.closeEntry();

			// add nn layout if present
			try {
				String layout = repository.loadLayout(name);
				zos.putNextEntry(new ZipEntry("layout.txt"));
				zos.write(layout.getBytes());
				zos.closeEntry();
			} catch (Exception e) {
			}

			// add weights in binary files
			try {
				Map<UUID, Tensor> weights = repository.loadParameters(name, tags);
				for (Entry<UUID, Tensor> e : weights.entrySet()) {
					String weightName = e.getKey().toString();
					if (tags != null && tags.length > 0) {
						for (String t : tags) {
							weightName += "-" + t;
						}
					}
					zos.putNextEntry(new ZipEntry(weightName));

					float[] data = e.getValue().get();
					dos.writeInt(data.length);
					for (int i = 0; i < data.length; i++) {
						dos.writeFloat(data[i]);
					}
					dos.flush();

					zos.closeEntry();
				}
			} catch (Exception e) {
				// ignore if no parameters available
			}

			zos.flush();
			zos.finish();
			baos.flush();
			return baos.toByteArray();
		}
	}

	private boolean isVersion(String v) {
		try {
			Version version = Version.parseVersion(v);
			return true;
		} catch (Exception e) {
			return false;
		}
	}

	private String dianneVersion() {
		Version v = context.getBundle().getVersion();
		// omit qualifier
		return v.getMajor() + "." + v.getMinor() + "." + v.getMicro();
	}
}
