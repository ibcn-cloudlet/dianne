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
import java.util.Map;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;

import be.iminds.iot.dianne.api.dataset.DianneDatasets;
import be.iminds.iot.dianne.api.dataset.Sequence;
import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;
import be.iminds.iot.dianne.tensor.util.JsonConverter;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/sb",
		 		 "osgi.http.whiteboard.servlet.pattern=/dianne/sb",
		 		 "osgi.http.whiteboard.servlet.asyncSupported:Boolean=true",
				 "aiolos.proxy=false" }, 
	immediate = true)
public class DianneSB extends HttpServlet {
	
	private static final long serialVersionUID = 1L;

	private JsonParser parser = new JsonParser();
	private JsonConverter converter = new JsonConverter();
	
	private Dianne dianne;
	private DiannePlatform platform;
	private DianneDatasets datasets;
	private Random random = new Random();
	
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
		
		ExperiencePool dataset = null;
		
		String d = request.getParameter("pool");
		if(d != null){
			dataset = (ExperiencePool)datasets.getDataset(d);
			if(dataset == null){
				System.out.println("Dataset "+d+" does not exist");
			}
		}

		String tag = request.getParameter("tag");
		
		NeuralNetwork encoder = null;
		String enc = request.getParameter("encoder");
		if(enc != null){
			encoder = nns.get(enc);
			if(encoder == null){
				try {
					encoder = dianne.getNeuralNetwork(platform.deployNeuralNetwork(enc, new String[]{tag})).getValue();
					nns.put(enc, encoder);
				} catch (Exception e) {
					e.printStackTrace();
					return;
				}
			} else {
				try {
					encoder.loadParameters(tag);
				} catch (Exception e) {
				}
			}
		}
		
		
		NeuralNetwork decoder = null;
		String dec = request.getParameter("decoder");
		if(dec != null){
			decoder = nns.get(dec);
			if(decoder == null){
				try {
					decoder = dianne.getNeuralNetwork(platform.deployNeuralNetwork(dec, new String[]{tag})).getValue();
					nns.put(dec, decoder);
				} catch (Exception e) {
					e.printStackTrace();
				}
			} else {
				try {
					decoder.loadParameters(tag);
				} catch (Exception e) {
				}
			}
		}

		NeuralNetwork predictor = null;
		String pre = request.getParameter("predictor");
		if(pre != null){
			predictor = nns.get(pre);
			if(predictor == null){
				try {
					predictor = dianne.getNeuralNetwork(platform.deployNeuralNetwork(pre, new String[]{tag})).getValue();
					nns.put(pre, predictor);
				} catch (Exception e) {
					e.printStackTrace();
				}
			} else {
				try {
					predictor.loadParameters(tag);
				} catch (Exception e) {
				}
			}
		} 

		NeuralNetwork rewardEstimator = null;
		String re = request.getParameter("reward");
		if(re != null){
			rewardEstimator = nns.get(re);
			if(rewardEstimator == null){
				try {
					rewardEstimator = dianne.getNeuralNetwork(platform.deployNeuralNetwork(re, new String[]{tag})).getValue();
					nns.put(re, rewardEstimator);
				} catch (Exception e) {
					e.printStackTrace();
				}
			} else {
				try {
					rewardEstimator.loadParameters(tag);
				} catch (Exception e) {
				}
			}
		} 
		
		
		int sequence = 0;
		String seq = request.getParameter("sequence");
		if(seq != null){
			sequence = Integer.parseInt(seq);
		}
		

		int index = 0;
		String in = request.getParameter("index");
		if(in != null){
			index = Integer.parseInt(in);
		}
		
		
		Tensor stateSample = null;
		String sa = request.getParameter("stateSample");
		if(sa!=null){
			stateSample = converter.fromJson(parser.parse(sa).getAsJsonObject());		
		}
		
		String sampleFrom = request.getParameter("sampleFrom");
		
		Tensor state = null;
		String s = request.getParameter("state");
		if(s!=null){
			state = converter.fromJson(parser.parse(s).getAsJsonObject());		
		} else {
			String sz = request.getParameter("stateSize");
			if(sz != null){
				state = new Tensor(Integer.parseInt(sz));
				state.fill(0.0f);
			} else if(stateSample == null){
				System.out.println("You should provide either state, stateSize or stateSample");
				return;
			}
		}
		
		boolean sampleReconstruction = true;
		if("false".equals(request.getParameter("sampleReconstruction"))){
			sampleReconstruction = false;
		}
		
		
		Tensor action = null;
		Tensor observation = null;
		Tensor reward = null;
		try {
			if(dataset != null){
				if(dataset.sequences() <= sequence){
					System.out.println("Sequence "+sequence+" does not exist for dataset "+d);
					return;
				}
				Sequence<ExperiencePoolSample> xp = dataset.getSequence(sequence);
				if(index > 0){
					if(index >= xp.size){
						System.out.println("Index "+index+" does not exist for sequence "+sequence+" in dataset "+d);
						return;
					}
					ExperiencePoolSample prev = xp.get(index-1);
					ExperiencePoolSample current = xp.get(index);
					action = prev.getAction();
					if(current.getState().size() == state.size()){
						// this is an xp pool with state samples... allows to inspect reconstructions
						stateSample = current.getState();
					} else {
						observation = current.getState();
					}
					reward = prev.getReward();
				} else {
					ExperiencePoolSample current = xp.get(0);
					action = current.getAction().clone();
					action.fill(0.0f);
					if(current.getState().size() == state.size()){
						// this is an xp pool with state samples... allows to inspect reconstructions
						stateSample = current.getState();
					} else {
						observation = current.getState();
					}
				}
			} else {
				String as = request.getParameter("actionSize");
				if(as != null){
					action = new Tensor(Integer.parseInt(as));
					action.fill(0.0f);
					if(index > 0){
						action.set(1.0f, random.nextInt(action.size()));
					}
				}
			}
			
			
			Tensor prior = null;
			if(predictor != null && action != null && state != null){
				if(index==0){
					state.fill(0.0f);
					action.fill(0.0f);
				}
				UUID[] pins = predictor.getModuleIds("State","Action");
				UUID[] pouts = predictor.getModuleIds("Output");
				prior = predictor.forward(pins, pouts, new Tensor[]{state, action}).getValue().tensor;
			}
			
			Tensor posterior = null;
			if(encoder != null && action != null && state != null && observation!=null){
				UUID[] eins = encoder.getModuleIds("State","Action","Observation");
				UUID[] eouts = encoder.getModuleIds("Output");
				posterior = encoder.forward(eins, eouts, new Tensor[]{state, action, observation}).getValue().tensor;
			}
			
			if(stateSample == null){
				if(prior != null && (posterior == null )|| "prior".equals(sampleFrom)){
					stateSample = sampleFromGaussian(prior);
				} else if(posterior != null){
					stateSample = sampleFromGaussian(posterior);
				}
			}
			
			if(stateSample == null){
				// error?!
				return;
			}
			
			Tensor reconstruction = null;
			if(decoder != null){
				Tensor reconstructionDistribution = decoder.forward(stateSample);
				if(reconstructionDistribution.dim() == 3){
					// softmax'ed!!!
					if(sampleReconstruction)
						reconstruction = sampleFromSoftmax(reconstructionDistribution);
					else
						reconstruction = reconstructionDistribution;
				} else {
					if(sampleReconstruction)
						reconstruction = sampleFromGaussian(reconstructionDistribution);
					else 
						reconstruction = reconstructionDistribution.narrow(0, 0, reconstructionDistribution.size()/2);
				}
			}
			
			Tensor rewardEstimate = null;
			if(rewardEstimator != null){
				UUID[] rins = rewardEstimator.getModuleIds("State","Action");
				UUID[] routs = rewardEstimator.getModuleIds("Output");
				
				rewardEstimate = rewardEstimator.forward(rins, routs, new Tensor[]{stateSample, action}).getValue().tensor;
			}
		
			JsonObject result = new JsonObject();
			if(state != null)
				result.add("state", converter.toJson(state));
			
			if(action != null)
				result.add("action", converter.toJson(action));
			
			if(observation != null)
				result.add("observation", converter.toJson(observation));
			
			if(prior != null)
				result.add("prior", converter.toJson(prior));
			
			if(posterior != null)
				result.add("posterior", converter.toJson(posterior));
			
			result.add("sample", converter.toJson(stateSample));
			
			if(reconstruction != null)
				result.add("reconstruction", converter.toJson(reconstruction));
		
			if(rewardEstimate != null)
				result.add("rewardEstimate", converter.toJson(rewardEstimate));
			
			if(reward != null)
				result.add("reward", new JsonPrimitive(reward.get(0)));
			
			response.getWriter().println(result);
			response.getWriter().flush();
		
		} catch(Exception e){
			e.printStackTrace();
		}
	}
	

	
	private Tensor sampleFromGaussian( Tensor distribution) {
		int size = distribution.size()/2;
		Tensor means = distribution.narrow(0, 0, size);
		Tensor stdevs = distribution.narrow(0, size, size);
		
		Tensor random = new Tensor(means.size());
		random.randn();
		
		Tensor sample = new Tensor(means.size());
		
		TensorOps.cmul(sample, random, stdevs);
		TensorOps.add(sample, sample, means);
		return sample;
	}
	
	
	private Tensor sampleFromSoftmax(Tensor distribution) {
		if(TensorOps.max(distribution) < 0){
			TensorOps.exp(distribution, distribution);
		}
		
		int size = distribution.dims()[2];
		Tensor sample = new Tensor(size);
		
		for(int i=0;i<size;i++){
			Tensor softmax = distribution.select(2, i);
			
			double s = 0, r = Math.random();
			int o = 0;
			while (o < softmax.size() && (s += softmax.get(o, 0)) < r) {
				o++;
			}
			
			sample.set(4*(float)o/softmax.size(), i);
		}
		
		return sample;
	}
}
