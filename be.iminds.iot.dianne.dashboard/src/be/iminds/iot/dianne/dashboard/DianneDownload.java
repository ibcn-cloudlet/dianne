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
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.nn.util.DianneJSONConverter;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(service={javax.servlet.Servlet.class},
property={"alias:String=/dianne/download",
	 	  "osgi.http.whiteboard.servlet.pattern=/dianne/download",
		  "aiolos.proxy=false"},
immediate=true)
public class DianneDownload extends HttpServlet{
	
	private static final long serialVersionUID = 1L;
	
	private DianneRepository repository;
	
	@Reference
	void setRepository(DianneRepository r){
		this.repository = r;
	}
	
	@Override
	protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
		String nnName = req.getParameter("nn");
		String tag = req.getParameter("tag");
		
		byte[] zip = zipNN(nnName, (tag == null || tag.isEmpty()) ? null : tag.split(","));

		if(zip != null){
			ServletOutputStream out = resp.getOutputStream();
			resp.setContentType("application/zip");
			resp.setHeader("Content-Disposition", "attachment; filename=\""+nnName+".jar\"");
	
			out.write(zip);
			out.flush();
		}
	}
	
	private byte[] zipNN(String nnName, String... tags) {
        try(
        	ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ZipOutputStream zos = new ZipOutputStream(baos);
        	DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(zos))
        ) {
        	// add manifest
        	Manifest manifest = new Manifest();
    		Attributes atts = manifest.getMainAttributes();
    		atts.put(Attributes.Name.MANIFEST_VERSION, "1.0");
    		atts.putValue(Constants.BUNDLE_MANIFESTVERSION, "2");
    		atts.putValue(Constants.BUNDLE_NAME, "Dianne NN "+nnName);
    		atts.putValue(Constants.BUNDLE_SYMBOLICNAME, "be.iminds.iot.dianne.nn."+nnName);
    		atts.putValue(Constants.BUNDLE_VERSION, "0.0.0");
    		atts.putValue("NeuralNetwork", nnName);
    		// TODO add requirement on DIANNE runtime?
    		
    		zos.putNextEntry(new ZipEntry("META-INF/MANIFEST.MF"));
    		manifest.write(zos);
    		zos.closeEntry();
        	
        	// add nn description
			NeuralNetworkDTO nn = repository.loadNeuralNetwork(nnName);
			String nnString = DianneJSONConverter.toJsonString(nn, true);
			
			zos.putNextEntry(new ZipEntry("modules.txt"));
			zos.write(nnString.getBytes());
			zos.closeEntry();
			
			// add nn layout if present
			try {
				String layout = repository.loadLayout(nnName);
				zos.putNextEntry(new ZipEntry("layout.txt"));
				zos.write(layout.getBytes());
				zos.closeEntry();
			} catch(Exception e){}

			// add weights in binary files
			Map<UUID, Tensor> weights = repository.loadParameters(nnName, tags);
			for(Entry<UUID, Tensor> e : weights.entrySet()){
				String weightName = e.getKey().toString();
				if(tags!=null && tags.length>0){
					for(String t : tags){
						weightName+="-"+t;
					}
				}
				zos.putNextEntry(new ZipEntry(weightName));
				
				float[] data = e.getValue().get();
				dos.writeInt(data.length);
				for(int i=0;i<data.length;i++){
					dos.writeFloat(data[i]);
				}
				dos.flush();
				
				zos.closeEntry();
			}
			
			zos.flush();
			zos.finish();
			baos.flush();
		    return baos.toByteArray();
        } catch(Exception e){
        }
        
        return null;
	}
}
