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
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;

import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

@Component(immediate=true)
public class DianneJSONRPCServer {

	private JSONRPCRequestHandler handler;
	
	private int port = 9090;
	private ServerSocket serverSocket;
	
	private Thread serverThread;
	
	
	private class JSONRPCHandler implements Runnable {
		
		private Socket socket;
		
		public JSONRPCHandler(Socket s){
			this.socket = s;
			try {
				this.socket.setKeepAlive(true);
			} catch (SocketException e) {
			}
		}
		
		public void run(){
			try {
				while(true){
					JsonWriter writer = new JsonWriter(new PrintWriter((socket.getOutputStream())));
					writer.flush();
					JsonReader reader = new JsonReader(new InputStreamReader(socket.getInputStream()));

					handler.handleRequest(reader, writer);
				}
			} catch(Exception e){
				try {
					socket.close();
				} catch (IOException e1) {
				}
			}
		}
		
		public void start(){
			Thread t = new Thread(this);
			t.start();
		}
		

	}

	
	@Activate
	void activate(BundleContext context) throws Exception {
		serverSocket = new ServerSocket(port);
		
		serverThread = new Thread(()->{
			while(!serverThread.isInterrupted()){
				try {
					Socket socket = serverSocket.accept();
					JSONRPCHandler handler = new JSONRPCHandler(socket);
					handler.start();
				} catch(Exception e){
					// e.printStackTrace();
				}
			}
		});
		serverThread.start();
	}
	
	@Deactivate
	void deactivate() throws Exception {
		serverThread.interrupt();
		serverSocket.close();
	}
	
	@Reference
	void setRequestHandler(JSONRPCRequestHandler h){
		this.handler = h;
	}
	
}
