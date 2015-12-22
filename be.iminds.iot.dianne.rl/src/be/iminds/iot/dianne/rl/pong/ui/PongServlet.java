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
package be.iminds.iot.dianne.rl.pong.ui;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.UnknownHostException;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

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
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;
import org.osgi.service.http.HttpService;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.api.rl.agent.Agent;
import be.iminds.iot.dianne.api.rl.environment.EnvironmentListener;
import be.iminds.iot.dianne.rl.agent.api.ManualActionController;
import be.iminds.iot.dianne.rl.pong.Pong;
import be.iminds.iot.dianne.rl.pong.api.PongEnvironment;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(service = { javax.servlet.Servlet.class, EnvironmentListener.class }, property = {
		"alias:String=/pong", "aiolos.proxy=false","target="+Pong.NAME }, immediate = true)
public class PongServlet extends HttpServlet implements EnvironmentListener {

	private DiannePlatform platform;

	// the Pong environment that is viewed
	private PongEnvironment pongEnvironment;

	// web socket server handling communication with UI clients
	private PongWebSocketServer pongWebSocket;

	private Agent agent;
	private ManualActionController agentAction;
	private TensorFactory factory;

	// for now hard coded
	private String nn = "DeepQPong";
	private String env = "Pong";
	private String pool = "Pong";
	private String strategy = "greedy";

	// interval between UI state updates
	private int interval = 20;
	private long timestamp = System.currentTimeMillis();

	@Activate
	void activate() {
		try {
			pongWebSocket = new PongWebSocketServer();
			pongWebSocket.start();
		} catch (UnknownHostException e) {
			System.err.println("Error starting Pong WebSocket server");
			e.printStackTrace();
		}
	}

	@Deactivate
	void deactivate() {
		try {
			pongWebSocket.stop();
		} catch (Exception e) {
			e.printStackTrace();
			// ignore
		}
	}

	@Reference
	void setHttpService(HttpService http) {
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
		resp.sendRedirect("http://" + req.getLocalAddr() + ":"
				+ req.getLocalPort() + "/pong/ui/pong.html");
	}

	@Reference
	void setPongEnvironment(PongEnvironment e) {
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
			conn.send("{" + "\"bounds\" : " + pongEnvironment.getBounds()
					+ ", \"paddleWidth\" : " + pongEnvironment.getPaddleWidth()
					+ ", \"paddleLength\" : "
					+ pongEnvironment.getPaddleLength() + ", \"ballRadius\" : "
					+ pongEnvironment.getBallRadius() + ", \"speed\" : "
					+ pongEnvironment.getSpeed() + "}");

		}

		@Override
		public void onMessage(WebSocket conn, String msg) {
			if (msg.startsWith("paction=")) {
				pongEnvironment.setOpponentAction(Integer.parseInt(msg
						.substring(8)));
			} else if (msg.startsWith("aaction=")) {
				float a = Integer.parseInt(msg.substring(8));
				float[] t = new float[] { a == 1 ? 1 : 0, a == 0 ? 1 : 0,
						a == -1 ? 1 : 0 };
				if (agentAction != null) {
					agentAction.setAction(factory.createTensor(t, 3));
				}
			} else if (msg.startsWith("ai=")) {
				if (msg.contains("human")) {
					pongEnvironment.useAI(false);
				} else {
					pongEnvironment.useAI(true);
				}
			} else if (msg.startsWith("agent=")) {
				strategy = msg.substring(6);
			} else if (msg.startsWith("interval=")) {
				int i = Integer.parseInt(msg.substring(9));
				interval = i;
			} else if (msg.startsWith("start")) {
				startAgent();
			}
		}

		@Override
		public void onError(WebSocket conn, Exception exception) {
			exception.printStackTrace();
		}

		@Override
		public void onClose(WebSocket conn, int code, String reason,
				boolean remote) {
		}

		public void sendToAll(String text) {
			Collection<WebSocket> con = connections();
			synchronized (con) {
				for (WebSocket c : con) {
					c.send(text);
				}
			}
		}
	}

	@Override
	public void onAction(float reward, Tensor nextState) {
		if (reward != 0) {
			pongWebSocket.sendToAll("{ \"score\" : " + reward + " }");
		}

		float[] data = nextState.get();
		pongWebSocket.sendToAll("{ "
				+ "\"x\" : " + data[0] + ", "
				+ "\"y\" : " + data[1] + ", "
				+ "\"vx\" : " + data[2] + ", "
				+ "\"vy\" : " + data[3] + ", "
				+ "\"p\" : " + data[4]+ ", "
				+ "\"o\" : " + data[5] + "}");

		long t = System.currentTimeMillis();
		long sleep = interval - (t - timestamp);
		timestamp = t;

		// since the PongListeners are called synchronously, slow it down here
		if (sleep > 0) {
			try {
				Thread.sleep(sleep);
			} catch (InterruptedException e) {
			}
		}
	}

	private void startAgent() {
		Map<String, String> config = new HashMap<String, String>();
		config.put("strategy", strategy);
		try {
			NeuralNetworkInstanceDTO nni = platform.deployNeuralNetwork("DeepQPong");
			this.agent.act(nni, env, pool, config);
		} catch (Exception e) {
			this.agent.stop();
		}
	}

	@Reference
	void setTensorFactory(TensorFactory f) {
		this.factory = f;
		if (agentAction != null) {
			this.agentAction.setAction(factory.createTensor(new float[] { 0, 1,
					0 }, 3));
		}
	}

	@Reference(cardinality = ReferenceCardinality.OPTIONAL, policy = ReferencePolicy.DYNAMIC)
	void setAgentAction(ManualActionController a) {
		this.agentAction = a;
		if (factory != null) {
			this.agentAction.setAction(factory.createTensor(new float[] { 0, 1,
					0 }, 3));
		}
	}

	public void unsetAgentAction(ManualActionController a) {
		if (this.agentAction == a) {
			this.agentAction = null;
		}
	}

	@Reference
	void setAgent(Agent a) {
		this.agent = a;
	}

	@Reference
	void setDiannePlatform(DiannePlatform p){
		this.platform = p;
	}
}
