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

import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import java.util.Map.Entry;
import java.util.UUID;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import javax.servlet.ServletException;
import javax.servlet.ServletOutputStream;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import com.google.gson.stream.JsonWriter;

import be.iminds.iot.dianne.api.coordinator.AgentResult;
import be.iminds.iot.dianne.api.coordinator.DianneCoordinator;
import be.iminds.iot.dianne.api.coordinator.EvaluationResult;
import be.iminds.iot.dianne.api.coordinator.Job;
import be.iminds.iot.dianne.api.coordinator.LearnResult;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.rl.agent.AgentProgress;
import be.iminds.iot.dianne.coordinator.util.DianneCoordinatorWriter;

@Component(service={javax.servlet.Servlet.class},
property={"alias:String=/dianne/results",
	 	  "osgi.http.whiteboard.servlet.pattern=/dianne/results",
		  "aiolos.proxy=false"},
immediate=true)
public class DianneResults extends HttpServlet{
	
	private static final long serialVersionUID = 1L;
	
	private DianneCoordinator coordinator;
	
	@Reference
	void setCoordinator(DianneCoordinator c){
		this.coordinator = c;
	}
	
	@Override
	protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
		UUID jobId = UUID.fromString(req.getParameter("jobId"));
		
		Job job = coordinator.getJob(jobId);

		if(job==null){
			System.out.println("Invalid job id "+jobId);
		}
		
		

		ServletOutputStream out = resp.getOutputStream();
		resp.setContentType("application/zip");
		resp.setHeader("Content-Disposition", "attachment; filename=\""+jobId+".zip\"");

		try (ZipOutputStream zos = new ZipOutputStream(out);
			 PrintWriter writer = new PrintWriter(zos);
			 JsonWriter json = new JsonWriter(writer);
		){
			json.setIndent("\t");
			
			zos.putNextEntry(new ZipEntry("job.json"));
			DianneCoordinatorWriter.writeJob(json, job);
			json.flush();
			zos.closeEntry();
			
			switch(job.type){
			case ACT:
				AgentResult ar = coordinator.getAgentResult(jobId);
				for(Entry<UUID, List<AgentProgress>> e : ar.progress.entrySet()){
					zos.putNextEntry(new ZipEntry(e.getKey()+".csv"));

					List<AgentProgress> progress = e.getValue();
					writer.println("sequence,iterations,reward(total),reward(last),terminal,epoch");
					progress.forEach(p -> {
						writer.print(p.sequence);
						writer.print(",");
						writer.print(p.iterations);
						writer.print(",");
						writer.print(p.reward);
						writer.print(",");
						writer.print(p.last);
						writer.print(",");
						writer.print(p.terminal);
						writer.print(",");
						writer.print(p.epoch);
						writer.println();
					});
					
					writer.flush();
					zos.closeEntry();
				}
				break;
			case LEARN:
				LearnResult lr = coordinator.getLearnResult(jobId);
				for(Entry<UUID, List<LearnProgress>> e : lr.progress.entrySet()){
					zos.putNextEntry(new ZipEntry(e.getKey()+".csv"));

					List<LearnProgress> progress = e.getValue();
					writer.print("iteration,minibatchLoss");
					if(progress.get(0).extra != null && !progress.get(0).extra.isEmpty()){
						progress.get(0).extra.entrySet()
							.forEach(extra -> {
									writer.print(",");
									writer.print(extra.getKey());
								});
					}
					writer.println();
					
					progress.forEach(p -> {
						writer.print(p.iteration);
						writer.print(",");
						writer.print(p.minibatchLoss);
						if(progress.get(0).extra != null && !progress.get(0).extra.isEmpty()){
							p.extra.entrySet()
								.forEach(extra -> {
										writer.print(",");
										writer.print(extra.getValue());
									});
						}
						writer.println();
					});
					
					writer.flush();
					zos.closeEntry();
				}
				break;
			case EVALUATE:
				EvaluationResult er = coordinator.getEvaluationResult(jobId);
				zos.putNextEntry(new ZipEntry("evaluation.json"));
				DianneCoordinatorWriter.writeEvaluationResult(json, er);
				json.flush();
				zos.closeEntry();
				break;
			}
		} catch(Exception e){
			e.printStackTrace();
		}
		
		out.flush();
	}
}
