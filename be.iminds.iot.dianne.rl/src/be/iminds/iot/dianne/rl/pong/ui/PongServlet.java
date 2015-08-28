package be.iminds.iot.dianne.rl.pong.ui;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.UnknownHostException;
import java.util.Collection;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.java_websocket.WebSocket;
import org.java_websocket.handshake.ClientHandshake;
import org.java_websocket.server.WebSocketServer;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.http.HttpService;

import be.iminds.iot.dianne.rl.pong.api.PongEnvironment;
import be.iminds.iot.dianne.rl.pong.api.PongListener;

@Component(service={javax.servlet.Servlet.class, PongListener.class},
	property={"alias:String=/pong","aiolos.proxy=false"},
	immediate=true)
public class PongServlet extends HttpServlet implements PongListener {

	// the Pong environment that is viewed
	private PongEnvironment pongEnvironment;
	
	// web socket server handling communication with UI clients
	private PongWebSocketServer pongWebSocket;
	
	@Activate
	public void activate(){
		try {
			pongWebSocket = new PongWebSocketServer();
			pongWebSocket.start();
		} catch (UnknownHostException e) {
			System.err.println("Error starting Pong WebSocket server");
			e.printStackTrace();
		}
	}
	
	@Deactivate
	public void deactivate(){
		try {
			pongWebSocket.stop();
		} catch (Exception e) {
			e.printStackTrace();
			// ignore
		} 
	}
	
	@Reference
	public void setHttpService(HttpService http){
		try {
			// TODO How to register resources with whiteboard pattern?
			http.registerResources("/pong/ui", "res", null);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	protected void doGet(HttpServletRequest req, HttpServletResponse resp)
			throws ServletException, IOException {
		resp.sendRedirect("pong/ui/pong.html");
	}
	
	@Reference
	public void setPongEnvironment(PongEnvironment e){
		this.pongEnvironment = e;
	}
	
	/**
	 * Inner class to handle WebSocket connections
	 * 
	 * @author tverbele
	 *
	 */
	class PongWebSocketServer extends WebSocketServer {

		public PongWebSocketServer() throws UnknownHostException {
			super(new InetSocketAddress(8787));
		}
		
		@Override
		public void onOpen(WebSocket conn, ClientHandshake handshake) {
			conn.send("{"
					+"\"bounds\" : "+ pongEnvironment.getBounds()
					+", \"paddleWidth\" : "+ pongEnvironment.getPaddleWidth()
					+", \"paddleLength\" : "+ pongEnvironment.getPaddleLength()
					+", \"ballRadius\" : "+ pongEnvironment.getBallRadius()
					+", \"speed\" : "+ pongEnvironment.getSpeed()
					+"}");
			
		}
		
		@Override
		public void onMessage(WebSocket conn, String msg) {
			// TODO how to determine who can control from web UI in case of multiple browsers?
			// TODO use button to flip AI vs manual control?
			pongEnvironment.useAI(false);
			pongEnvironment.setOpponentAction(Integer.parseInt(msg));
		}
		
		@Override
		public void onError(WebSocket conn, Exception exception) {
			exception.printStackTrace();
		}

		
		@Override
		public void onClose(WebSocket conn,  int code, String reason, boolean remote) {
		}

		public void sendToAll( String text ) {
			Collection<WebSocket> con = connections();
			synchronized ( con ) {
				for( WebSocket c : con ) {
					c.send( text );
				}
			}
		}
	}

	@Override
	public void update(float x, float y, float vx, float vy, float p, float o) {
		pongWebSocket.sendToAll("{ \"x\" : "+x
							   +", \"y\" : "+y
							   +", \"vx\" : "+vx
							   +", \"vy\" : "+vy
							   +", \"p\" : "+p
							   +", \"o\" : "+o
							   +"}");
		
		// since the PongListeners are called synchronously, slow it down here
		try {
			Thread.sleep(20);
		} catch (InterruptedException e) {
		}
	}
}
