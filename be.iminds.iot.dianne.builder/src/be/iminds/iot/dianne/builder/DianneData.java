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
package be.iminds.iot.dianne.builder;

import java.io.IOException;
import java.util.Random;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.DatasetDTO;
import be.iminds.iot.dianne.api.dataset.DianneDatasets;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.tensor.TensorOps;
import be.iminds.iot.dianne.tensor.util.JsonConverter;


@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/datasets",
		 		 "osgi.http.whiteboard.servlet.pattern=/dianne/datasets",
				 "aiolos.proxy=false" }, 
	immediate = true)
public class DianneData extends HttpServlet {

	private static final long serialVersionUID = 1L;
	
	private Random rand = new Random(System.currentTimeMillis());
	private JsonConverter converter = new JsonConverter();
	
	private DianneDatasets datasets;
	
	@Reference
	void setDianneDatasets(DianneDatasets d){
		this.datasets = d;
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		response.setContentType("application/json");
		String action = request.getParameter("action");
		
		if(action.equals("available-datasets")){
			JsonArray result = new JsonArray();
			synchronized(datasets){
				for(DatasetDTO d : datasets.getDatasets()){
					JsonObject r = new JsonObject();
					r.add("dataset", new JsonPrimitive(d.name));
					r.add("size", new JsonPrimitive(d.size));
					if(d.inputType != null)
						r.add("inputType", new JsonPrimitive(d.inputType));
					if(d.targetType != null)
						r.add("targetType", new JsonPrimitive(d.targetType));
					String[] ll = d.labels;
					if(ll != null){
						JsonArray labels = new JsonArray();
						for(String l :ll){
							labels.add(new JsonPrimitive(l));
						}
						r.add("labels", labels);
					}
					result.add(r);
				}
			}
			response.getWriter().println(result.toString());
			response.getWriter().flush();
		} else if(action.equals("sample")){
			String dataset = request.getParameter("dataset");
			Dataset d = datasets.getDataset(dataset);
			if(d!=null && d.size() > 0){
				int index = rand.nextInt(d.size());
				Sample s = d.getSample(index);
				JsonObject sample = converter.toJson(s.input);
				sample.add("index", new JsonPrimitive(index));
				String[] labels = d.getLabels();
				if(labels != null){
					sample.add("target", new JsonPrimitive(labels[TensorOps.argmax(s.target)]));
				} else {
					if(s.target.size() < 10){
						JsonObject target = converter.toJson(s.target);
						sample.add("target", target.get("data").getAsJsonArray());
					}
				}
				
				response.getWriter().println(sample.toString());
				response.getWriter().flush();
			}
		}
	}
}
