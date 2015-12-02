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
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.tensor.Tensor;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;


@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/datasets",
		 		 "osgi.http.whiteboard.servlet.pattern=/dianne/datasets",
				 "aiolos.proxy=false" }, 
	immediate = true)
public class DianneDatasets extends HttpServlet {

	private Random rand = new Random(System.currentTimeMillis());
	private JsonParser parser = new JsonParser();
	
	private Map<String, Dataset> datasets = Collections.synchronizedMap(new HashMap<String, Dataset>());

	@Reference(cardinality=ReferenceCardinality.AT_LEAST_ONE, 
			policy=ReferencePolicy.DYNAMIC)
	void addDataset(Dataset dataset, Map<String, Object> properties){
		String name = (String) properties.get("name");
		this.datasets.put(name, dataset);
	}
	
	void removeDataset(Dataset dataset, Map<String, Object> properties){
		String name = (String) properties.get("name");
		datasets.remove(name);
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		
		String action = request.getParameter("action");
		
		if(action.equals("available-datasets")){
			JsonArray result = new JsonArray();
			synchronized(datasets){
				for(Dataset d : datasets.values()){
					JsonObject r = new JsonObject();
					r.add("dataset", new JsonPrimitive(d.getName()));
					r.add("size", new JsonPrimitive(d.size()));
					JsonArray labels = new JsonArray();
					for(String l : d.getLabels()){
						labels.add(new JsonPrimitive(l));
					}
					r.add("labels", labels);
					result.add(r);
				}
			}
			response.getWriter().println(result.toString());
			response.getWriter().flush();
		} else if(action.equals("sample")){
			String dataset = request.getParameter("dataset");
			Dataset d = datasets.get(dataset);
			if(d!=null){
				JsonObject sample = new JsonObject();
				
				Tensor t = d.getInputSample(rand.nextInt(d.size()));
				
				if(t.dims().length==3){
					sample.add("channels", new JsonPrimitive(t.dims()[0]));
					sample.add("height", new JsonPrimitive(t.dims()[1]));
					sample.add("width", new JsonPrimitive(t.dims()[2]));
				} else {
					sample.add("channels", new JsonPrimitive(1));
					sample.add("height", new JsonPrimitive(t.dims()[0]));
					sample.add("width", new JsonPrimitive(t.dims()[1]));
				}
				sample.add("data", parser.parse(Arrays.toString(t.get())));
				response.getWriter().println(sample.toString());
				response.getWriter().flush();
			}
		}
	}
}
