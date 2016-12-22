

function render(tensor, canvasCtx, type){
	console.log("RENDER "+type)
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
		// batched inputs ... try to fit a mosaic
		if(tensor.dims.length == 4){
			var batchSize = tensor.dims[0];
			var mosaic = Math.ceil(Math.sqrt(batchSize));
			var mosaicW = canvasW/mosaic;
			var mosaicH = canvasH/mosaic;
			
			var offset = 0;
			tensor.dims.splice(0, 1);
			for(l=0;l<mosaic;l++){
				for(k=0;k<mosaic;k++){
					if(offset >= tensor.size)
						continue;
					
					image_rect(tensor, canvasCtx, offset, k*mosaicW, l*mosaicH, mosaicW, mosaicH);
					offset = offset + tensor.size/batchSize;
				}
			}
			
		}
		
	} else {
		image_rect(tensor, canvasCtx, 0, 0, 0, canvasW, canvasH);
	}
}

function image_rect(tensor, canvasCtx, offset, posX, posY, imageW, imageH){
	canvasCtx.clearRect(posX,posY,imageW,imageH);

	var w = tensor.dims[tensor.dims.length-1];
	var h = tensor.dims[tensor.dims.length-2];
	if(h === undefined)
		h = 1;
	
	var scaleX = imageW/w;
	var scaleY = imageH/h;
	var scale = scaleX < scaleY ? scaleX : scaleY;
	
	var width = Math.round(w*scale);
	var height = Math.round(h*scale);
	var channels = tensor.dims.length > 2 ? tensor.dims[tensor.dims.length-3] : 1;
	var imageData = canvasCtx.createImageData(width, height);
	
	
	// render single image
	if(channels===1){
		for (var y = 0; y < height; y++) {
	        for (var x = 0; x < width; x++) {
	        	// collect alpha values
	        	var x_s = Math.floor(x/scale);
	        	var y_s = Math.floor(y/scale);
	        	var index = offset + y_s*w+x_s;
	        	imageData.data[y*width*4+x*4+3] = Math.floor(tensor.data[index]*255);
	        }
	    }
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
	}	
	
	var offsetX = Math.floor((imageW-width)/2);
	var offsetY = Math.floor((imageH-height)/2);
	canvasCtx.putImageData(imageData, posX+offsetX, posY+offsetY); 
}


function laser(tensor, canvasCtx, showTarget){
	var canvasW = canvasCtx.canvas.clientWidth;
	var canvasH = canvasCtx.canvas.clientHeight;
	
	// render laserdata
	var step = Math.PI/tensor.size;
	var angle = 0;
	var length;
	canvasCtx.clearRect(0,0,canvasW,canvasH);
	canvasCtx.beginPath();
	for (var i = 0; i < tensor.size; i++) {
		canvasCtx.moveTo(canvasW/2, canvasH);
		length = tensor.data[i]*canvasH/2;
		canvasCtx.lineTo(canvasW/2+length*Math.cos(angle), canvasH-length*Math.sin(angle));
		angle+=step;
	}
	canvasCtx.stroke();
	canvasCtx.closePath();
	
	if(showTarget){
		canvasCtx.beginPath();
		canvasCtx.arc(256, 449, 6, 0, 2 * Math.PI, false);
		canvasCtx.fillStyle = 'red';
		canvasCtx.fill();
		canvasCtx.closePath();
	}
}