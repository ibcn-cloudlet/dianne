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
import java.util.Comparator;
import java.util.Map;
import java.util.Random;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.DianneDatasets;
import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;
import be.iminds.iot.dianne.tensor.util.JsonConverter;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/vae",
		 		 "osgi.http.whiteboard.servlet.pattern=/dianne/vae",
		 		 "osgi.http.whiteboard.servlet.asyncSupported:Boolean=true",
				 "aiolos.proxy=false" }, 
	immediate = true)
public class DianneVAE extends HttpServlet {
	
	private static final long serialVersionUID = 1L;

	private Random random = new Random(System.currentTimeMillis());
	
	private JsonConverter converter = new JsonConverter();
	
	private Dianne dianne;
	private DiannePlatform platform;
	private DianneDatasets datasets;
	
	private int tries = 1000;
	private int size = 6*6;
	
	private Map<String, NeuralNetwork> nns = new ConcurrentHashMap<>();
	
	@Reference
	void setDianne(Dianne d){
		dianne = d;
	}
	
	@Reference
	void setDiannePlatform(DiannePlatform p){
		platform = p;
	}
	
	@Reference(cardinality=ReferenceCardinality.OPTIONAL)
	void setDianneDatasets(DianneDatasets d){
		datasets = d;
	}
	
	@Override
	protected void doGet(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {

		
		
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		response.setContentType("application/json");
		String d = request.getParameter("dataset");
		if(d == null){
			System.out.println("No dataset provided");
			return;
		}
		Dataset dataset = datasets.getDataset(d);
		if(dataset == null){
			System.out.println("No such dataset found: "+d);
			return;
		}
		
		
		String enc = request.getParameter("encoder");
		if(enc == null){
			System.out.println("No encoder provided");
			return;
		}
		
		NeuralNetwork encoder = nns.get(enc);
		if(encoder == null){
			try {
				encoder = dianne.getNeuralNetwork(platform.deployNeuralNetwork(enc, new String[]{"vae"})).getValue();
				nns.put(enc, encoder);
			} catch (Exception e) {
				e.printStackTrace();
				return;
			}
		}
		
		
		NeuralNetwork decoder = null;
		String dec = request.getParameter("decoder");
		if(dec != null){
			decoder = nns.get(dec);
			if(decoder == null){
				try {
					decoder = dianne.getNeuralNetwork(platform.deployNeuralNetwork(dec, new String[]{"vae"})).getValue();
					nns.put(dec, decoder);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}
		
		int size = this.size;
		String s = request.getParameter("size");
		if(s != null){
			size = Integer.parseInt(s);
		}
		
		JsonArray result = new JsonArray();

		String sample = request.getParameter("sample");
		String l = request.getParameter("latent");
		if(sample == null && l == null){
			// fetch random samples from dataset
			for(int i=0;i<size;i++){
				JsonObject o = new JsonObject();
				int index = random.nextInt(dataset.size());
				Tensor data = dataset.getSample(index).input;
				Tensor latent = encoder.forward(data);
				
				o.add("index", new JsonPrimitive(index));
				o.add("data", converter.toJson(data));
				o.add("latent", converter.toJson(latent));

				result.add(o);
			}
			
		} else {
			Tensor latent = null;
			if(sample != null){
				int index = Integer.parseInt(sample);
				Tensor base = dataset.getSample(index).input;
				latent = encoder.forward(base).clone();
			} else {
				Tensor lmeans = converter.fromString(l);
				latent = new Tensor(lmeans.size()*2);
				latent.fill(0.0f);
				lmeans.copyInto(latent.narrow(0, lmeans.size()));
			}
			if(decoder != null){
				// generate samples using the decoder
				Tensor mean = latent.narrow(0, latent.size()/2);
				Tensor stdev = latent.narrow(latent.size()/2, latent.size()/2);
				Tensor rand = new Tensor(latent.size()/2);
				Tensor rand2 = null;
				
				for(int i=0;i<size;i++){
					rand.randn();
					TensorOps.cmul(rand, rand, stdev);
					TensorOps.add(rand, rand, mean);
					
					Tensor reconstruction = decoder.forward(rand);
					Tensor rmean = reconstruction.narrow(0, reconstruction.size()/2);
					Tensor rstdev = reconstruction.narrow(reconstruction.size()/2, reconstruction.size()/2);
					
					if(rand2 == null){
						rand2 = new Tensor(rmean.dims());
					}
					rand2.randn();
					TensorOps.cmul(rand2, rand2, rstdev);
					TensorOps.add(rand2, rand2, rmean);
					
					JsonObject o = new JsonObject();
					o.add("index", new JsonPrimitive(-1));
					o.add("data", converter.toJson(rand2));
					o.add("latent", converter.toJson(latent));
					result.add(o);
				}
				
			} else {
				// search for samples close to this sample in the dataset
				final Tensor baseline = latent;
				SortedSet<LatentSample> set = new TreeSet<>(new Comparator<LatentSample>() {
					@Override
					public int compare(LatentSample o1, LatentSample o2) {
						Tensor diff = TensorOps.sub(null, o1.latent.narrow(0, baseline.size()/2), baseline.narrow(0, baseline.size()/2));
						Float d1 = TensorOps.dot(diff, diff);
						diff = TensorOps.sub(null, o2.latent.narrow(0, baseline.size()/2), baseline.narrow(0, baseline.size()/2));
						Float d2 = TensorOps.dot(diff, diff);
						return d1.compareTo(d2);
					}
				});
				
				for(int i=0;i<tries;i++){
					LatentSample ls = new LatentSample();
					ls.index = random.nextInt(dataset.size());
					ls.data = dataset.getSample(ls.index).input;
					ls.latent = encoder.forward(ls.data).clone();
					set.add(ls);
					
					if(set.size() > size){
						set.remove(set.last());
					}
				}
				
				for(LatentSample ls : set){
					JsonObject o = new JsonObject();
					o.add("index", new JsonPrimitive(ls.index));
					o.add("data", converter.toJson(ls.data));
					o.add("latent", converter.toJson(ls.latent));
					result.add(o);
				}
			}
		}
		
		response.getWriter().println(result);
		response.getWriter().flush();
	}
	
	private class LatentSample {
		public int index;
		public Tensor data;
		public Tensor latent;
	}
}
