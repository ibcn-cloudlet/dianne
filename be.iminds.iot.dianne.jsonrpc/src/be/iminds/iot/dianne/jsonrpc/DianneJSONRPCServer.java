package be.iminds.iot.dianne.jsonrpc;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.util.Map;
import java.util.stream.Collectors;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.coordinator.DianneCoordinator;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.nn.util.DianneJSONConverter;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

@Component(immediate=true)
public class DianneJSONRPCServer {

	private DianneCoordinator coordinator;
	
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
					
					// read input
					JsonParser parser = new JsonParser();
					JsonObject request = parser.parse(reader).getAsJsonObject();

					System.out.println("REQUEST "+request.toString());
					if(!request.has("jsonrpc")){
						System.out.println("Invalid JSONRPC request");
						return;
					}
					
					if(!request.get("jsonrpc").getAsString().equals("2.0")){
						System.out.println("Wrong JSONRPC version: "+request.get("jsonrpc").getAsString());
						return;
					}
					
					if(!request.has("method")){
						System.out.println("No method specified");
						return;
					}

					String method = request.get("method").getAsString();

					
					String i = "null";
					if(request.has("id")){
						i = request.get("id").getAsString();
					}
					final String id = i;
					
					
					// for now only supported method
					if(method.equals("learn")){
						NeuralNetworkDTO nn;
						String dataset;
						Map<String, String> config;
						
						try {
							JsonArray params = request.get("params").getAsJsonArray();
							nn = DianneJSONConverter.parseJSON(params.get(0).getAsJsonObject());
							dataset = params.get(1).getAsString();
							config = params.get(2).getAsJsonObject()
									.entrySet().stream().collect(Collectors.toMap( e -> e.getKey(), e -> e.getValue().getAsString()));

						} catch(Exception e){
							System.out.println("Incorrect parameters provided");
							return;
						}
						
						// call coordinator
						coordinator.learn(nn, dataset, config).then(p -> {

							// write output when promise resolves
							writer.beginObject();
							writer.name("jsonrpc");
							writer.value("2.0");
							writer.name("id");
							writer.value(id);
							writer.endObject();
							writer.flush();
							
							return null;
						});
						
					}
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
	void setDianneCoordinator(DianneCoordinator c){
		this.coordinator = c;
	}
	
}
