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
package be.iminds.iot.dianne.dashboard;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Map;
import java.util.Map.Entry;
import java.util.UUID;
import java.util.jar.Attributes;
import java.util.jar.Manifest;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import javax.servlet.ServletException;
import javax.servlet.ServletOutputStream;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.framework.Constants;
import org.osgi.framework.Version;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.util.DianneExporter;
import be.iminds.iot.dianne.nn.util.DianneJSONConverter;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(service={javax.servlet.Servlet.class},
property={"alias:String=/dianne/download",
	 	  "osgi.http.whiteboard.servlet.pattern=/dianne/download",
		  "aiolos.proxy=false"},
immediate=true)
public class DianneDownload extends HttpServlet{
	
	private static final long serialVersionUID = 1L;
	
	private DianneExporter exporter;
	
	@Reference
	void setExporter(DianneExporter e){
		this.exporter = e;
	}
	
	@Override
	protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
		String nnName = req.getParameter("nn");
		String tag = req.getParameter("tag");
		
		byte[] zip = exporter.export(nnName, (tag == null || tag.isEmpty()) ? null : tag.split(","));

		if(zip != null){
			ServletOutputStream out = resp.getOutputStream();
			resp.setContentType("application/zip");
			resp.setHeader("Content-Disposition", "attachment; filename=\""+nnName+".jar\"");
	
			out.write(zip);
			out.flush();
		}
	}
}
