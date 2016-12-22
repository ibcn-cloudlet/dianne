

function render(tensor, canvasCtx, type){
	if(type==="image"){
		image(tensor, canvasCtx);
	} else if(type==="laser"){
		laser(tensor, canvasCtx, false);
	} else {
		// image by default?
		image(tensor, canvasCtx);
	}
}


function image(tensor, canvasCtx){
	var canvasW = canvasCtx.canvas.clientWidth;
	var canvasH = canvasCtx.canvas.clientHeight;
	
	if(tensor.dims.length >= 4){
		// batched inputs ... render a mosaic
		if(tensor.dims.length == 4){
			var batchSize = tensor.dims[0];
			var mosaic = Math.ceil(Math.sqrt(batchSize));
			if(height == 1){
				var mosaicW = canvasW;
				var mosaicH = canvasH/(mosaic*mosaic)
			} else {
				var mosaicW = canvasW/mosaic;
				var mosaicH = canvasH/mosaic;
			}
			
			var offset = 0;
			tensor.dims.splice(0, 1);
			var i = 0;
			for(l=0;l<mosaic;l++){
				for(k=0;k<mosaic;k++){
					if(offset >= tensor.size)
						continue;
					
					image_rect(tensor, canvasCtx, offset, height == 1 ? 0 : k*mosaicW,  h == 1 ? (i++)*mosaicH : l*mosaicH, mosaicW, mosaicH);
					offset = offset + tensor.size/batchSize;
				}
			}
		}
	} else {
		image_rect(tensor, canvasCtx, 0, 0, 0, canvasW, canvasH);
	}
}


function image_rect(tensor, canvasCtx, offset, posX, posY, targetW, targetH){
	canvasCtx.clearRect(posX,posY,targetW,targetH);

	var w = tensor.dims[tensor.dims.length-1];
	var h = tensor.dims[tensor.dims.length-2];
	if(h === undefined)
		h = 1;
	
	var scaleX = targetW/w;
	var scaleY = targetH/h;
	var scale = scaleX < scaleY ? scaleX : scaleY;
	
	var width = Math.round(w*scale);
	var height = Math.round(h*scale);
	var channels = tensor.dims.length > 2 ? tensor.dims[tensor.dims.length-3] : 1;
	var imageData = canvasCtx.createImageData(width, h==1 ? targetH : height);
	
	if(channels===1){
		for (var y = 0; y < height; y++) {
	        for (var x = 0; x < width; x++) {
	        	// collect alpha values
	        	var x_s = Math.floor(x/scale);
	        	var y_s = Math.floor(y/scale);
	        	var index = offset + y_s*w+x_s;
	        	var normalized = tensor.data[index];
	        	if(tensor.min < 0 || tensor.max > 1){
	        		normalized = (normalized - tensor.min)/(tensor.max-tensor.min);
	        	}
	        	
	        	if(h == 1){
	        		for(var yy = 0 ; yy < targetH; yy++){
	        			imageData.data[yy*width*4+x*4+3] = Math.floor(normalized*255);
	        		}
	        	} else {
        			imageData.data[y*width*4+x*4+3] = Math.floor(normalized*255);
	        	}
	        }
	    }
		
		var offsetX = h==1 ? 0 : Math.floor((targetW-width)/2);
		var offsetY = h==1 ? 1 : Math.floor((targetH-height)/2);
		canvasCtx.putImageData(imageData, posX+offsetX, posY+offsetY); 
	} else if(channels===3){
		// RGB
		for(var c = 0; c < 3; c++){
			for (var y = 0; y < height; y++) {
		        for (var x = 0; x < width; x++) {
		        	var x_s = Math.floor(x/scale);
		        	var y_s = Math.floor(y/scale);
		        	var index = offset + c*w*h + y_s*w+x_s;
		        	imageData.data[y*width*4+x*4+c] = Math.floor(tensor.data[index]*255);
		        }
		    }		
		}
		for (var y = 0; y < height; y++) {
	        for (var x = 0; x < width; x++) {
	        	imageData.data[y*width*4+x*4+3] = 255;
	        }
		}
		
		var offsetX = Math.floor((targetW-width)/2);
		var offsetY = Math.floor((targetH-height)/2);
		canvasCtx.putImageData(imageData, posX+offsetX, posY+offsetY); 
	} else {
		// render each channel as mosaic image
		var mosaic = Math.ceil(Math.sqrt(channels));
		if(h == 1){
			var mosaicW = targetW;
			var mosaicH = targetH/(mosaic*mosaic)
		} else {
			var mosaicW = targetW/mosaic;
			var mosaicH = targetH/mosaic;
		}
		
		tensor.dims.splice(0, 1);
		var i = 0;
		for(l=0;l<mosaic;l++){
			for(k=0;k<mosaic;k++){
				if(offset >= tensor.size)
					continue;
				
				image_rect(tensor, canvasCtx, offset, h == 1 ? 0 : k*mosaicW, h == 1 ? (i++)*mosaicH : l*mosaicH, mosaicW, mosaicH);
				offset = offset + tensor.size/channels;
			}
		}
	}
}


function laser(tensor, canvasCtx, showTarget){
	var canvasW = canvasCtx.canvas.clientWidth;
	var canvasH = canvasCtx.canvas.clientHeight;
	
	if(tensor.dims.length >= 2){
		// render in a mosaic
		var scanPoints = tensor.dims[tensor.dims.length-1];
		var batchSize = tensor.size/scanPoints;
		var mosaic = Math.ceil(Math.sqrt(batchSize));
		var mosaicW = canvasW/mosaic;
		var mosaicH = canvasH/mosaic;
		
		var offset = 0;
		for(l=0;l<mosaic;l++){
			for(k=0;k<mosaic;k++){
				if(offset >= tensor.size)
					continue;
				
				laser_rect(tensor, canvasCtx, offset, scanPoints, k*mosaicW, l*mosaicH, mosaicW, mosaicH, showTarget);
				offset = offset + scanPoints;
			}
		}
	} else {
		laser_rect(tensor, canvasCtx, 0, tensor.size, 0, 0, canvasW, canvasH, showTarget);
	}
}


function laser_rect(tensor, canvasCtx, offset, scanPoints, posX, posY, targetW, targetH, showTarget){
	// render laserdata
	var step = Math.PI/scanPoints;
	var angle = 0;
	var length;
	canvasCtx.clearRect(posX,posY,targetW,targetH);
	canvasCtx.beginPath();
	for (var i = 0; i < scanPoints; i++) {
		canvasCtx.moveTo(posX+targetW/2, posY+targetH);
		length = tensor.data[offset+i]*targetH/2;
		
		var x = posX+targetW/2+length*Math.cos(angle);
		var y = posY+targetH-length*Math.sin(angle);
		canvasCtx.lineTo(Math.min(Math.max(parseInt(x), posX), posX+targetW), 
				Math.min(Math.max(parseInt(y), posY), posY+targetH));
		angle+=step;
	}
	canvasCtx.stroke();
	canvasCtx.closePath();
	
	if(showTarget){
		canvasCtx.beginPath();
		canvasCtx.arc(targetW/2, 0.876953125*targetH, 6, 0, 2 * Math.PI, false);
		canvasCtx.fillStyle = 'red';
		canvasCtx.fill();
		canvasCtx.closePath();
	}
}