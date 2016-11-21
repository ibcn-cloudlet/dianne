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
package be.iminds.iot.dianne.jsonrpc;

import java.io.IOException;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

@Component(service = { javax.servlet.Servlet.class }, property = { "alias:String=/dianne/jsonrpc",
		"osgi.http.whiteboard.servlet.pattern=/dianne/jsonrpc", "aiolos.proxy=false" }, immediate = true)
public class DianneJSONRPCServlet extends HttpServlet {

	private static final long serialVersionUID = 1L;

	private JSONRPCRequestHandler handler;

	@Reference
	void setRequestHandler(JSONRPCRequestHandler h) {
		this.handler = h;
	}

	@Override
	protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
		resp.setContentType("application/json");
		
		try (JsonReader reader = new JsonReader(req.getReader());
				JsonWriter writer = new JsonWriter(resp.getWriter())) {

			handler.handleRequest(reader, writer);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
