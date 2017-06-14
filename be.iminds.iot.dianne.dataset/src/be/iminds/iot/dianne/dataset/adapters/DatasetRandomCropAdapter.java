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
package be.iminds.iot.dianne.dataset.adapters;

import java.util.Map;
import java.util.Random;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * This Dataset adapter extends the Dataset by taking random crops of the image
 * 
 * Configure by providing a width and height of the target crop. These can also be ranges 
 * in which the width and height can vary. e.g.
 * 
 * cropWidth = 32, cropHeight = 32  will calculate random 32x32 crops
 * cropWidth = [32,64] cropHeight = [32,64] will calculate random crops with width and height between 32 and 64
 * 
 * Optionally also a cropPadding parameter can be provided that adds zero padding to the original 
 * sample before cropping.
 * 
 * @author tverbele
 *
 */
@Component(
	service={Dataset.class},
	configurationPolicy=ConfigurationPolicy.REQUIRE,
	configurationPid="be.iminds.iot.dianne.dataset.adapters.RandomCropAdapter")
public class DatasetRandomCropAdapter extends AbstractDatasetAdapter {

	// resulting width should be between minWidth and maxWidth
	private int minWidth;
	private int maxWidth;
	// resulting height should be between minHeight and maxHeight;
	private int minHeight;
	private int maxHeight;
	// add additional padding
	private int padding = 0;
	
	private Random r = new Random(System.currentTimeMillis());
	
	protected void configure(Map<String, Object> properties) {
		Object w = properties.get("cropWidth");
		if(w instanceof String[]){
			minWidth = Integer.parseInt(((String[]) w)[0].trim());
			maxWidth = Integer.parseInt(((String[]) w)[1].trim());
		} else {
			minWidth = Integer.parseInt((String)w);
			maxWidth = minWidth;
		}
		
		Object h = properties.get("cropHeight");
		if(h instanceof String[]){
			minHeight = Integer.parseInt(((String[]) h)[0].trim());
			maxHeight = Integer.parseInt(((String[]) h)[1].trim());
		} else {
			minHeight = Integer.parseInt(((String)h).trim());
			maxHeight = minHeight;
		}
		
		if(properties.containsKey("cropPadding")){
			padding = Integer.parseInt(((String) properties.get("cropPadding")).trim());
		}
	}
	
	@Override
	public int[] inputDims(){
		if(minWidth != maxWidth || minHeight != maxHeight)
			return null;
		
		int[] dims = null;
		int[] originalDims = data.inputDims();
		if(originalDims == null){
			// assume 3d images
			dims = new int[]{3, maxHeight, maxWidth};
		} else {
			dims = new int[originalDims.length];
			if(originalDims.length == 3){
				dims[0] = originalDims[0];
				dims[1] = maxHeight;
				dims[2] = maxWidth;
			} else {
				dims[0] = maxHeight;
				dims[1] = maxWidth;
			}
		}
		
		return dims;
	}
	
	@Override
	public int[] targetDims(){
		if(!targetDimsSameAsInput){
			return data.targetDims();
		} else {
			return inputDims();
		}
	}
	
	@Override
	protected void adaptSample(Sample original, Sample adapted) {
		int width = (int)Math.floor(minWidth + r.nextFloat()*(maxWidth-minWidth));
		int height = (int)Math.floor(minHeight + r.nextFloat()*(maxHeight-minHeight));
		
		// create random crop with widhtxheight
		int[] dims = original.input.dims();
		
		int originalHeight = dims.length == 3 ? dims[1] : dims[0];
		int originalWidth = dims.length == 3 ? dims[2] : dims[1];
		
		// if original dims are smaller than requested crop, scale up first
		float scale = 1;
		while(originalHeight < height || originalWidth < width){
			originalHeight*=2;
			originalWidth*=2;
			scale *= 2;
		}
		
		if(scale > 1){
			if(dims.length == 3){
				dims[1] = originalHeight;
				dims[2] = originalWidth;
			} else {
				dims[0] = originalHeight;
				dims[1] = originalWidth;
			}
			original.input = TensorOps.scale2D(null, original.input, dims);
		}
		
		// calculate random offsets
		int originalWidthOffset = (int)Math.floor((originalWidth+2*padding - width)*r.nextFloat());
		int originalHeightOffset = (int)Math.floor((originalHeight+2*padding - height)*r.nextFloat());
	
		// crop
		adapted.input = crop(adapted.input, original.input, width, height, originalWidthOffset, originalHeightOffset);
		
		// if target is also image - also crop target
		if(targetDimsSameAsInput){
			adapted.target = crop(adapted.target, original.target, width, height, originalWidthOffset, originalHeightOffset);
		} else {
			adapted.target = original.target.copyInto(adapted.target);
		}
	}
	
	
	Tensor crop(Tensor res, final Tensor t, int width, int height, int widthOffset, int heightOffset){
		int[] dims = t.dims();
		int channels = dims.length == 3 ? dims[0] : 1;

		if(res == null){
			res = new Tensor(channels, height, width);
		} else {
			res.reshape(channels, height, width);
		}

		if(padding > 0){
			res.fill(0.0f);
			
			int adaptedWidthOffset = widthOffset < padding ? padding - widthOffset : 0;
			int adaptedHeightOffset = heightOffset < padding ? padding - heightOffset : 0;
			int originalWidthOffset = widthOffset >= padding ? widthOffset - padding : widthOffset;
			int originalHeightOffset = heightOffset >= padding ? heightOffset - padding : heightOffset;
	
			int cropWidth = originalWidthOffset > adaptedWidthOffset ? width - originalWidthOffset : width - adaptedWidthOffset; 
			int cropHeight = originalHeightOffset > adaptedHeightOffset ? height - originalHeightOffset : height - adaptedHeightOffset; 
	
			int[] originalRanges = new int[]{0, channels, originalHeightOffset, cropHeight, originalWidthOffset, cropWidth};
			int[] adaptedRanges = new int[]{0, channels, adaptedHeightOffset, cropHeight, adaptedWidthOffset, cropWidth};
			
			Tensor dst = padding > 0 ? res.narrow(adaptedRanges) : res;
			Tensor src = t.narrow(originalRanges);
			src.copyInto(dst);
		
		} else {
			Tensor src = t.narrow(0, channels, heightOffset, height, widthOffset, width);
			src.copyInto(res);
		}
		
		return res;
	}
}
