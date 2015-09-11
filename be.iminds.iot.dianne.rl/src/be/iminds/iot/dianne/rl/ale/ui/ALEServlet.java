package be.iminds.iot.dianne.rl.ale.ui;

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

import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.http.HttpService;

import be.iminds.iot.dianne.api.rl.EnvironmentListener;
import be.iminds.iot.dianne.rl.ale.ArcadeLearningEnvironment;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.dianne.tensor.util.ImageConverter;

@Component(service = { javax.servlet.Servlet.class, EnvironmentListener.class }, property = {
		"alias:String=/ale", "aiolos.proxy=false","target="+ArcadeLearningEnvironment.NAME }, immediate = true)
public class ALEServlet extends HttpServlet implements EnvironmentListener {

	// interval between UI state updates
	private int interval = 17;
	private long timestamp = System.currentTimeMillis();
	
	private Map<String, CameraStream> streams = new HashMap<>();
	private TensorFactory factory;
	private ImageConverter converter;

	@Activate
	public void activate(){
		converter = new ImageConverter(factory);
	}
	
	@Reference
	public void setHttpService(HttpService http) {
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
			stream = new CameraStream(client);
			synchronized(streams){
				streams.put(client, stream);
			}
		}
		
		stream.updateRequest(request);

	}
	
	@Override
	public void onAction(float reward, Tensor nextState) {
		try {
			BufferedImage img = converter.writeToImage(nextState);
			
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
						streams.remove(s.getClient());
						it.remove();
						s.close();
					}
				}
			}
		
		} catch(Exception e){
			// ignore
		}
		
		
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
	
	private class CameraStream {
		
		private final String client;
		
		private AsyncContext async;
		private ServletResponse response; 
		
		public CameraStream(String client){
			this.client = client;
		}
		
		protected void sendFrame(byte[] data) throws IOException {
			response.getOutputStream().println("--next");
			response.getOutputStream().println("Content-Type: image/jpeg");
			response.getOutputStream().println("Content-Length: "+data.length);
			response.getOutputStream().println("");
			response.getOutputStream().write(data, 0, data.length);
			response.getOutputStream().println("");
			response.flushBuffer();
		}
		
		protected String getClient(){
			return client;
		}
		
		protected void updateRequest(ServletRequest request){
			this.async = request.startAsync();
			this.response = async.getResponse();
		}
		
		protected void close(){
			this.async.complete();
		}
	}
	
	@Reference
	void setTensorFactory(TensorFactory factory) {
		this.factory = factory;
	}
}
