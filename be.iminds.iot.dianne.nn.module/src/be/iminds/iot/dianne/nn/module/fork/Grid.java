package be.iminds.iot.dianne.nn.module.fork;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class Grid extends AbstractModule {

	// crop last two dimensions to y,x
	private final int x;
	private final int y;
	private final int stride_x;
	private final int stride_y;
	// for now don't use scale factor, but could be used in the future?
	private final float scale = 1;
	
	private Map<String, Tensor> crops = new HashMap<String, Tensor>();
	
	public Grid(TensorFactory factory, int x, int y, int sx, int sy) {
		super(factory);
		this.x = x;
		this.y = y;
		this.stride_x = sx;
		this.stride_y = sy;
	}
	
	public Grid(TensorFactory factory, UUID id, int x, int y, int sx, int sy) {
		super(factory, id);
		this.x = x;
		this.y = y;
		this.stride_x = sx;
		this.stride_y = sy;	
	}

	@Override
	protected void callNext(){
		// call next for each crop
		for(String tag : crops.keySet()){
			int l = tags.length;
			String[] t = Arrays.copyOf(tags, l+1);
			t[l] = tag;
			
			runExecutor.execute(new ForwardRunnable(next[0], crops.get(tag), t));
		}
	}
	
	
	@Override
	protected void forward() {
		// generate crops using target size and strides and forward those separately
		
		crops.clear();
		int x_dim = input.dim()-1;
		int y_dim = x_dim-1;
		
		int size_x = input.dims()[x_dim];
		int size_y = input.dims()[y_dim];
		
		// if size_x or size_y <  x,y ... then scale up
		float s = scale;
		if(size_x < x){
			float ratio = ((float)x)/size_x;
			s = s > ratio ? s : ratio;
		}
		if(size_y < y){
			float ratio = ((float)y)/size_y;
			s = s > ratio ? s : ratio;
		}
		
		Tensor scaled;
		if(scale!=1){
			int[] scaledDims = Arrays.copyOf(input.dims(), input.dim());
			scaledDims[x_dim] = (int)(size_x*s);
			scaledDims[y_dim] = (int)(size_y*s);
			scaled = factory.getTensorMath().scale2D(null, scaled, scaledDims);
		} else {
			scaled = input;
		}
		
		int scaled_x = scaled.dims()[x_dim];
		int scaled_y = scaled.dims()[y_dim];
		
		int crop_x = (scaled_x-x)/stride_x+1;
		int crop_y = (scaled_y-y)/stride_y+1;
		
		for(int i=0;i<crop_x;i++){
			for(int j=0;j<crop_y;j++){
				String tag = "Grid_"+i+"_"+j; // TODO make unique, e.g. use UUID?

				Tensor crop = scaled.narrow(y_dim, j*stride_y, y);
				crop = crop.narrow(x_dim, i*stride_x, x);
				
				crops.put(tag, crop);
			}
		}
		
	}

	@Override
	protected void backward() {
		// not to be used in training?
		throw new UnsupportedOperationException();
	}
	
}
