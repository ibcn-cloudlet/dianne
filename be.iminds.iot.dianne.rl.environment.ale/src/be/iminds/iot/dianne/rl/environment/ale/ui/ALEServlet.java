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
package be.iminds.iot.dianne.rl.environment.ale.ui;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import javax.imageio.ImageIO;
import javax.imageio.stream.ImageOutputStream;
import javax.servlet.AsyncContext;
import javax.servlet.ServletException;
import javax.servlet.ServletRequest;
import javax.servlet.ServletResponse;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.http.HttpService;

import be.iminds.iot.dianne.api.rl.environment.EnvironmentListener;
import be.iminds.iot.dianne.rl.environment.ale.ArcadeLearningEnvironment;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.util.ImageConverter;

@Component(service = { javax.servlet.Servlet.class, EnvironmentListener.class }, property = {
		"alias:String=/ale", "aiolos.proxy=false","target="+ArcadeLearningEnvironment.NAME }, immediate = true)
public class ALEServlet extends HttpServlet implements EnvironmentListener {

	private static final long serialVersionUID = 1L;

	// interval between UI state updates
	private int interval = 17;
	private long timestamp = System.currentTimeMillis();
	
	private Map<String, CameraStream> streams = new HashMap<>();
	private ImageConverter converter;
	private boolean grayscale = true;

	@Activate
	void activate(BundleContext context){
		converter = new ImageConverter();
		
    	String gray = context.getProperty("be.iminds.iot.dianne.rl.ale.grayscale");
    	if(gray!=null){
    		grayscale = Boolean.parseBoolean(gray);
    	}
	}
	
	@Reference
	void setHttpService(HttpService http) {
		try {
			// TODO How to register resources with whiteboard pattern?
			http.registerResources("/ale/ui", "res", null);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	protected void doGet(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		String mjpeg = request.getParameter("mjpeg");
		
		if(mjpeg==null){
			response.sendRedirect("http://" + request.getLocalAddr() + ":"
					+ request.getLocalPort() + "/ale/ui/ale.html");
			return;
		}
		

		response.setHeader("Cache-Control", "no-cache");
		response.setCharacterEncoding("UTF-8");
		response.addHeader("Connection", "keep-alive");
		response.setContentType("multipart/x-mixed-replace;boundary=next");

		// check if there is already a stream for this client
		String client = request.getRemoteHost()+":"+request.getRemotePort();
		CameraStream stream = streams.get(client);
		if(stream==null){
			stream = new CameraStream();
			synchronized(streams){
				streams.put(client, stream);
			}
		}
		
		stream.updateRequest(request);

	}
	
	@Override
	public void onAction(float reward, Tensor nextState) {
		try {
			BufferedImage img = null;
			if(grayscale){
				img = converter.writeToImage(nextState.select(0, nextState.size(0)-1));
			} else {
				img = converter.writeToImage(nextState.narrow(0, nextState.size(0)-4, 3));
			}
			
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			ImageOutputStream ios = ImageIO.createImageOutputStream(bos);
			ImageIO.write(img, "jpg",ios);
			byte[] jpeg = bos.toByteArray();
			
	
			synchronized(streams){
				Iterator<CameraStream> it = streams.values().iterator();
				while(it.hasNext()){
					CameraStream s = it.next();
					try {
						s.sendFrame(jpeg);
					} catch(IOException e){
						it.remove();
						s.close();
					}
				}
			}
		
		} catch(Exception e){
			// ignore
			e.printStackTrace();
		}
		
		
		long t = System.currentTimeMillis();
		long sleep = interval - (t - timestamp);
		timestamp = t;

		if (sleep > 0) {
			try {
				Thread.sleep(sleep);
			} catch (InterruptedException e) {
			}
		}
	}
	
	private class CameraStream {
		
		private AsyncContext async;
		private ServletResponse response; 
		
		protected void sendFrame(byte[] data) throws IOException {
			response.getOutputStream().println("--next");
			response.getOutputStream().println("Content-Type: image/jpeg");
			response.getOutputStream().println("Content-Length: "+data.length);
			response.getOutputStream().println("");
			response.getOutputStream().write(data, 0, data.length);
			response.getOutputStream().println("");
			response.flushBuffer();
		}
		
		protected void updateRequest(ServletRequest request){
			this.async = request.startAsync();
			this.response = async.getResponse();
		}
		
		protected void close(){
			this.async.complete();
		}
	}
	
}
