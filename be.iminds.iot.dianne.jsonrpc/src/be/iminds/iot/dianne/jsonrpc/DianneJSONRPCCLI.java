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

import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.URI;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=jsonrpc"},
		immediate=true)
public class DianneJSONRPCCLI {

	private JSONRPCRequestHandler handler;
	private JsonParser parser = new JsonParser();
	
	@Reference
	void setRequestHandler(JSONRPCRequestHandler h) {
		this.handler = h;
	}
	
	public void jsonrpc(String requestString) {
		JsonWriter writer = new JsonWriter(new PrintWriter(System.out));
		try {
			if(requestString.contains("://")){
				// treat as URI
				URI uri = new URI(requestString);
				JsonReader reader = new JsonReader(new InputStreamReader(uri.toURL().openStream()));
				handler.handleRequest(reader, writer);
			} else {
				// treat as json request string
				JsonObject request = parser.parse(requestString).getAsJsonObject(); 
				handler.handleRequest(request, writer);
			}
		} catch(Exception e){
			e.printStackTrace();
		} 
	}
}
