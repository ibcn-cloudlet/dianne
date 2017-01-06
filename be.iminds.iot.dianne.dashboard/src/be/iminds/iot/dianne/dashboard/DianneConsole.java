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

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.felix.service.command.CommandProcessor;
import org.apache.felix.service.command.CommandSession;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

@Component(service={javax.servlet.Servlet.class},
property={"alias:String=/dianne/console",
	 	  "osgi.http.whiteboard.servlet.pattern=/dianne/console",
		  "aiolos.proxy=false"},
immediate=true)
public class DianneConsole extends HttpServlet{

	private static final long serialVersionUID = 1L;

	private CommandSession session;

	private ByteArrayOutputStream output;
	private ByteArrayOutputStream error;
	private PrintStream outputStream;
	private PrintStream errorStream;
	
	@Override
	protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		
		String command = request.getParameter("command");
		
		String result = "";
		try {
			session.execute(command);
		} catch (Exception e) {
			e.printStackTrace(errorStream);
		}
		
		if(output.size() > 0){
			result = new String(output.toByteArray(), StandardCharsets.UTF_8);
		} else if(error.size() > 0){
			result = new String(error.toByteArray(), StandardCharsets.UTF_8);
		}
		
		output.reset();
		error.reset();
		
		response.getWriter().print(result);
		response.getWriter().flush();
	}
	
	@Reference
	void setCommandProcessor(CommandProcessor cp){
		output = new ByteArrayOutputStream();
		error = new ByteArrayOutputStream();

		outputStream= new PrintStream(output);
		errorStream = new PrintStream(error);
		this.session = cp.createSession(System.in, outputStream, errorStream);
	}
}
