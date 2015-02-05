package be.iminds.iot.dianne.builder;

import java.io.IOException;
import java.io.PrintWriter;

import javax.servlet.AsyncContext;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.http.HttpService;

import be.iminds.iot.dianne.nn.module.AbstractModule;
import be.iminds.iot.dianne.nn.module.ForwardListener;
import be.iminds.iot.dianne.nn.module.Input;
import be.iminds.iot.dianne.nn.module.Output;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/run","aiolos.proxy=false" }, 
	immediate = true)
public class DianneRunner extends HttpServlet {
	
	private TensorFactory factory;
	
	// for now fixed 1 input, 1 output, and trainable modules
	private Input input;
	private Output output;

	private AsyncContext sse = null;
	
	@Reference
	public void setTensorFactory(TensorFactory factory){
		this.factory = factory;
	}
	
	@Reference
	public void setInput(Input input){
		this.input = input;
	}
	
	@Reference
	public void setOutput(Output output){
		this.output = output;
		
		((AbstractModule)output).addForwardListener(new ForwardListener() {
			@Override
			public void onForward(Tensor output) {
				if(sse!=null){
					try {
						StringBuilder builder = new StringBuilder();
						builder.append("data: ").append(output.toString()).append("\n\n");
		
						PrintWriter writer = sse.getResponse().getWriter();
						writer.write(builder.toString());
						writer.flush();
					} catch(Exception e){}
				}
			}
		});
	}
	
	@Reference
	public void setHttpService(HttpService http){
		try {
			// Use manual registration - problems with whiteboard
			http.registerServlet("/dianne/run", this, null, null);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	@Override
	protected void doGet(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		// write text/eventstream response
		response.setContentType("text/event-stream");
		response.setHeader("Cache-Control", "no-cache");
		response.setCharacterEncoding("UTF-8");
		response.addHeader("Connection", "keep-alive");
		
		sse = request.startAsync();
		sse.setTimeout(0); // no timeout => remove listener when error occurs.
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		float[] data = parseInput(request.getParameter("forward"));
		Tensor t = factory.createTensor(data, 28, 28);
		input.input(t);
	}
	

	private float[] parseInput(String string){
		String[] strings = string.replace("[", "").replace("]", "").split(",");
		float result[] = new float[strings.length];
		for (int i = 0; i < result.length; i++) {
			result[i] = Float.parseFloat(strings[i]);
		}
		return result;
	}
}
