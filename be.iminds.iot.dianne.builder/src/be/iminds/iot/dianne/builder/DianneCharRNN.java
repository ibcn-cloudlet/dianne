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
import java.io.PrintWriter;
import java.util.UUID;

import javax.servlet.AsyncContext;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/charrnn",
				 "osgi.http.whiteboard.servlet.pattern=/dianne/output",
				 "aiolos.proxy=false" }, 
	immediate = true)
public class DianneCharRNN extends HttpServlet {
	
	private static final long serialVersionUID = 1L;
	
	private Dianne dianne;
	private DiannePlatform platform;
	
	@Reference
	void setDianne(Dianne d){
		dianne = d;
	}
	
	@Reference
	void setDiannePlatform(DiannePlatform p){
		platform = p;
	}
	
	@Override
	protected void doGet(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		// write text/eventstream response
		response.setContentType("text/event-stream");
		response.setHeader("Cache-Control", "no-cache");
		response.setCharacterEncoding("UTF-8");
		response.addHeader("Connection", "keep-alive");
		
		AsyncContext async = request.startAsync();
		PrintWriter writer = async.getResponse().getWriter();
		
		String id = request.getParameter("id");
		if(id == null){
			System.out.println("No neural network instance specified");
			return;
		}
		UUID nnId = UUID.fromString(id);
		NeuralNetworkInstanceDTO nni = platform.getNeuralNetworkInstance(nnId);
		if(nni==null){
			System.out.println("Neural network instance "+id+" not deployed");
			return;
		}
		
		NeuralNetwork nn = null;
		try {
			nn = dianne.getNeuralNetwork(nni).getValue();
		} catch (Exception e) {
		}
		if(nn==null){
			System.out.println("Neural network instance "+id+" not available");
			return;
		}
		
		int size = 1;
		String charsequence = "";
		
		if(request.getParameter("size")!=null){
			size = Integer.parseInt(request.getParameter("size"));
		}
		
		if(request.getParameter("charsequence")!=null){
			charsequence = request.getParameter("charsequence");
		}
		
		for(int i=0;i<charsequence.length()-1;i++){
			nextChar(nn, charsequence.charAt(i));
		}
		
		char c = '\n'; // if no charsequence given start from newline?
		if(charsequence.length() > 0)
			c = charsequence.charAt(charsequence.length()-1);
		
		int chunksize = size > 1000 ? 100 : 1;
		
		String chunk = "";
		for(int i=0;i<size;i++){
			c = nextChar(nn, c);
			
			if(c == '\n'){
				chunk += "<br>";
			} else {
				chunk += c;
			}
			
			if(chunk.length() >= chunksize || i==size-1){
				StringBuilder builder = new StringBuilder();
				builder.append("data: ").append(chunk).append("\n\n");
				writer.write(builder.toString());
				writer.flush();
				chunk = "";
			}
		}
		
		async.complete();
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
	}
	
	private char nextChar(NeuralNetwork nn, char current){
		// construct input tensor
		String[] labels = nn.getOutputLabels();
		if(labels==null){
			throw new RuntimeException("Neural network "+nn.getNeuralNetworkInstance().name+" is not trained and has no labels");
		}
		Tensor in = new Tensor(labels.length);
		in.fill(0.0f);
		int index = 0;
		for(int i=0;i<labels.length;i++){
			if(labels[i].charAt(0)==current){
				index = i;
				break;
			}
		}
		if(index < labels.length){
			in.set(1.0f, index);
		}
		
		// forward
		Tensor out = nn.forward(in);
		
		// select next, sampling from (Log)Softmax output
		if(TensorOps.min(out) < 0){
			// assume logsoftmax output, take exp
			out = TensorOps.exp(out, out);
		}
		
		double s = 0, r = Math.random();
		int o = 0;
		while(o < out.size() && (s += out.get(o)) < r){
			o++;
		}
		
		return labels[o].charAt(0);
	}
}
