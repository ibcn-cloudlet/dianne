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
package be.iminds.iot.dianne.coordinator.util;

import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import com.google.gson.stream.JsonWriter;

import be.iminds.iot.dianne.api.coordinator.AgentResult;
import be.iminds.iot.dianne.api.coordinator.EvaluationResult;
import be.iminds.iot.dianne.api.coordinator.Job;
import be.iminds.iot.dianne.api.coordinator.LearnResult;
import be.iminds.iot.dianne.api.nn.eval.ClassificationEvaluation;
import be.iminds.iot.dianne.api.nn.eval.ErrorEvaluation;
import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.eval.EvaluationProgress;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.rl.agent.AgentProgress;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * Helper class for writing Jobs and their results to JSONWriter... used in coordinator and jsonrpc
 * 
 * @author tverbele
 *
 */
public class DianneCoordinatorWriter {

	public static void writeObject(JsonWriter writer, Object o) throws Exception {
		if(o==null){
			writer.value("null");
		} else if(o instanceof LearnResult){
			writeLearnResult(writer, (LearnResult) o);
		} else if(o instanceof EvaluationResult){
			writeEvaluationResult(writer, (EvaluationResult) o);
		} else if(o instanceof AgentResult){
			writeAgentResult(writer, (AgentResult) o);
		} else if(o.getClass().equals(String.class)
				|| o.getClass().isEnum()
				|| o.getClass().equals(UUID.class)){
			writer.value(o.toString());
		} else if(o instanceof List){
			List<?> l = (List<?>)o;
			writer.beginArray();
			for(Object ll : l){
				writeObject(writer, ll);
			}
			writer.endArray();
		} else if(o instanceof Map){
			Map<?,?> m = (Map<?,?>) o;
			writer.beginObject();
			for(Object k : m.keySet()){
				writer.name(k.toString());
				writeObject(writer, m.get(k));
			}
			writer.endObject();
		}else if(o.getClass().isArray()){ 
			int length = Array.getLength(o);
			writer.beginArray();
			for (int i = 0; i < length; i++) {
				writeObject(writer, Array.get(o, i));
			}
			writer.endArray();
		} else {
			writer.beginObject();
			writeFields(writer, o);
			writer.endObject();
		}
	}
	
	public static void writeFields(JsonWriter writer, Object o) throws Exception {
		for(Field f : o.getClass().getFields()){
			if(Modifier.isPublic(f.getModifiers())){
				writer.name(f.getName());
				if(f.getType().isPrimitive()){ 
					switch(f.getType().getName()){
					case "long":
						writer.value(f.getLong(o));
						break;
					case "int":
						writer.value(f.getInt(o));
						break;
					case "float":
						writer.value(f.getFloat(o));
						break;
					case "double":
						writer.value(f.getDouble(o));
						break;
					case "boolean":
						writer.value(f.getBoolean(o));
						break;
					case "short": 
						writer.value(f.getShort(o));
						break;
					case "byte":
						writer.value(f.getByte(o));
						break;
					}
				} else {
					writeObject(writer, f.get(o));
				}
			}
		}
	}
	
	public static void writeJob(JsonWriter writer, Job job) throws Exception {
		// job is just an ordinary object
		writeObject(writer, job);
	}
	
	public static void writeLearnResult(JsonWriter writer, LearnResult result) throws Exception {
		writer.beginArray();
		// merge progress and validation in single object
		// TODO for now select one learners minibatch loss as progress?
		if(result.progress.size() > 0){
			List<LearnProgress> select = result.progress.values().iterator().next();
			for(int i =0;i<select.size();i++){
				LearnProgress p = select.get(i);
				Evaluation val = result.validations.get(p.iteration);
				
				writer.beginObject();
				writeFields(writer, p);
				if(val != null){
					writer.name("validationLoss");
					writer.value(val.metric());
				}
				writer.endObject();
			}
		}
		writer.endArray();
	}
	
	public static void writeEvaluationResult(JsonWriter writer, EvaluationResult result) throws Exception {
		writer.beginArray();
		for(Evaluation eval : result.evaluations.values()){
			writer.beginObject();
			if(eval==null){
				// write nothing?
			} else if(eval instanceof EvaluationProgress){
				writer.name("processed");
				writer.value(((EvaluationProgress) eval).processed());
				writer.name("total");
				writer.value(((EvaluationProgress) eval).size());
				writer.name("metric");
				writer.value(((EvaluationProgress) eval).metric());
			} else {
				writer.name("evaluationTime");
				writer.value(eval.time());
				
				if(eval instanceof ErrorEvaluation){
					ErrorEvaluation eeval = (ErrorEvaluation)eval;
					
					writer.name("error");
					writer.value(new Float(eeval.error()));

					writer.name("forwardTime");
					writer.value(new Float(eeval.forwardTime()));
					
					// write all outputs
					if(eeval.outputs()!=null){
						writer.name("outputs");
						writer.beginArray();
						for(Tensor t : eeval.outputs()){
							writer.beginArray();
							for(float f : t.get()){
								writer.value(new Float(f));
							}
							writer.endArray();
						}
						writer.endArray();
					}
				}
				
				if(eval instanceof ClassificationEvaluation){
					ClassificationEvaluation ceval = (ClassificationEvaluation) eval;
					// write accuracy
					writer.name("accuracy");
					writer.value(new Float(ceval.accuracy()));
					
					writer.name("top3");
					writer.value(new Float(ceval.topNaccuracy(3)));

					writer.name("top5");
					writer.value(new Float(ceval.topNaccuracy(5)));

					// write confusion matrix
					writer.name("confusionMatrix");
					writer.beginArray();
					Tensor confusionMatrix = ceval.confusionMatrix();
					for(int i=0;i<confusionMatrix.size(0);i++){
						writer.beginArray();
						for(int j=0;j<confusionMatrix.size(1);j++){
							writer.value(new Float(confusionMatrix.get(i, j)));
						}
						writer.endArray();
					}
					writer.endArray();
				}
			} 
			writer.endObject();	
		}
		writer.endArray();
	}
	
	public static void writeAgentResult(JsonWriter writer, AgentResult result) throws Exception {
		writer.beginArray();
		for(List<AgentProgress> pp : result.progress.values()){
			writer.beginArray();
			for(int i =0;i<pp.size();i++){
				AgentProgress progress = pp.get(i);
				writer.beginObject();
				writeFields(writer, progress);
				writer.endObject();
			}
			writer.endArray();
		}
		writer.endArray();
	}
	
}
