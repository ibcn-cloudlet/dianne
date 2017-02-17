/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  Ghent University, iMinds
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
/*
 * Module configuration/deletion dialog stuff 
 */
 
 var dialogZIndex = 1040;

/**
 * Show a dialog for a given module, will forward to right function depending on currentMode
 */
function showConfigureModuleDialog(moduleItem) {
	var id = moduleItem.attr("id");
	// there can be only one dialog at a time for one module
	// try to reuse dialog
	var dialogId = "dialog-" + id;
	var dialog;
	dialog = $("#" + dialogId);
	if (dialog.length == 0) {
		
		if (currentMode === "build") {
			dialog = createBuildModuleDialog(id, moduleItem);
		} else if (currentMode === "deploy") {
			dialog = createDeployModuleDialog(id, moduleItem);
		} else if (currentMode === "learn") {
			dialog = createLearnModuleDialog(id, moduleItem);
		} else if (currentMode === "run") {
			dialog = createRunModuleDialog(id, moduleItem);
		}
		
		if (dialog !== undefined) {
			var offset = moduleItem.offset();
			offset.top = offset.top - 100;
			offset.left = offset.left - 200;
			// show the modal (disable backdrop)
			dialog.draggable({
				handle : ".modal-header"
			}).mousedown(function(){
	   			// set clicked element to a higher level
	   			$(this).css('z-index', ++dialogZIndex);
			}).offset(offset);
		}
	}
	
	if (dialog !== undefined) {
		dialog.modal({
			'show' : true,
			'backdrop' : false
		}).css('z-index', ++dialogZIndex);
	}
}

/*
 * Helper function to create base dialog and show the Module div
 * Base for each NN configure module dialog in each mode
 */
function createNNModuleDialog(module, title, submit, cancel){
	var dialog = renderTemplate("dialog", {
		'id' : module.id,
		'type': 'config',
		'title' : title,
		'submit': submit,
		'cancel': cancel
	}, $(document.body));
	
	// add module div to dialog to show which module to configure
	renderTemplate("module",
			{	
				name: module.name,
				type: module.type, 
				category: module.category
			}, 
			dialog.find('.content'));
	dialog.find('.content').append('<div class="inline">'+module.type+' - '+module.id+'</div>');
	
	return dialog;
}

/**
 * Create dialog for configuring module in build mode
 */
function createBuildModuleDialog(id, moduleItem){
	var module = nn.modules[id];
	
	var dialog = createNNModuleDialog(module, "Configure module ", "Configure", "Delete");
	
	// then fill in properties
	$.post("/dianne/builder", {"action" : "module-properties","type" : module.type}, 
			function( data ) {
				renderTemplate('form-item',
					{
						name: 'Name',
						id: 'name',
						value: module.name
					}, dialog.find('.form-items'));
				
				$.each(data, function(index, property){
					// Render toolbox item
					renderTemplate('form-item',
						{
							name: property.name,
							id: property.id,
							value: module[property.id]
						}, dialog.find('.form-items'));
				});
				if (data.length === 0) {
					dialog.find('.form-items').append("<p>No properties to configure...</p>");
				}
			}
			, "json");
	
	// set button callbacks, disable buttons when module is deployed
	if(deployment[id]!==undefined){
		dialog.find(".submit").prop('disabled', true);
		dialog.find(".cancel").prop('disabled', true);
	} else {
		dialog.find(".submit").click(function(e){
			// apply configuration
			var data = $(this).closest('.modal').find('form').serializeArray();
			
			var module;
			$.each( data, function( i, item ) {
				if(i === 0){
					module = nn.modules[item.value];
				} else {
					module[item.name] = item.value;
				}
			});
			
			// update name
			var moduleItem = $('#'+id);
			moduleItem.attr('name', module.name);
			moduleItem.html('<strong>'+module.name+'</strong>');
			
			$(this).closest(".modal").modal('hide');
		});
		
		dialog.find(".cancel").click(function(e){
			// remove object
			var id = $(this).closest(".modal").find(".module-id").val();
			
			var moduleItem = $('#'+id);
			if(checkRemoveModule(moduleItem)) {
				removeModule(moduleItem);
			}
			
			// remove dialog when module is removed, else keep it for reuse
			$(this).closest(".modal").remove();
		});
	}
	
	return dialog;
}


/**
 * Create dialog for configuring module in deploy mode
 */
function createDeployModuleDialog(id, moduleItem){
	var module = nn.modules[id];
	
	var dialog = createNNModuleDialog(module, "Deploy module ", "Deploy", "Undeploy");
	
	// fill in deployment options
	if(deployment[id]===undefined){
		renderTemplate("form-dropdown", 
				{	
					name: "Deploy to: "
				},
				dialog.find('.form-items'));
		$.post("/dianne/deployer", {"action" : "targets"}, 
				function( data ) {
					$.each(data, function(index, target){
						dialog.find('.options').append("<option value="+target.id+">"+target.name+"</option>")
					});
				}
				, "json");

	} else {
		dialog.find('.form-items').append("<p>This module is deployed to "+deployment[id]+"</p>");
	}
	
	// add button callbacks
	if(deployment[id]===undefined){
		dialog.find(".submit").click(function(e){
			// deploy this module
			var id = $(this).closest(".modal").find(".module-id").val();
			var target = $(this).closest('.modal').find('.options').val();
			
			deploy(id, target);
			
			$(this).closest(".modal").remove();
		});
		dialog.find(".cancel").prop('disabled', true);
	} else {
		dialog.find(".cancel").click(function(e){
			// undeploy this module
			var id = $(this).closest(".modal").find(".module-id").val();
			undeploy(id);
			
			$(this).closest(".modal").remove();
		});
		dialog.find(".submit").prop('disabled', true);
	}
	
	return dialog;
}


/**
 * Create dialogs for learning modules
 */
function createLearnModuleDialog(id, moduleItem){
	var module = learning[id];
	if(module===undefined){
		module = nn.modules[id];
		
		if(module.category==="Fork"
					|| module.category==="Join"){
			
			var dialog = createNNModuleDialog(module, "Configure module", "Save", "");
			dialog.find(".cancel").remove();

			renderTemplate("form-dropdown", 
				{	
					name: "Mode"
				},
				dialog.find('.form-items'));
			
			dialog.find('.options').append("<option value=\"FORWARD_ON_CHANGE\">Forward on change</option>");
			dialog.find('.options').append("<option value=\"WAIT_FOR_ALL\">Wait for all input/gradOutput</option>");
			dialog.find('.options').change(function(event){
				var selected = dialog.find( "option:selected" ).val();
				var id = dialog.find(".module-id").val();

				// weird to do this with run, but actually makes sense to set runtime mode in run servlet?
				$.post("/dianne/run", {"mode":selected, "target":id, "id": nn.id}, 
						function( data ) {
						}
						, "json");
				
				$(this).closest(".modal").modal('hide');
			});
			
			return dialog;
		} else {
			// no dialogs for untrainable modules
			return undefined;
		}
	}
	
	var dialog;
	if(module.category==="Dataset"){
		dialog = renderTemplate("dialog", {
			id : id,
			type: "dataset",
			title : "Configure "+module.type+" dataset",
			submit: "",
			cancel: "Delete"
		}, $(document.body));
		
		renderTemplate("dataset-learn", {
				id : module.id,
				dataset : module.dataset,
				train: module.train,
				test: module.test,
				validation: module.validation
			},
			dialog.find('.content')
		);
		
		dialog.find(".slider").slider({
			orientation: "vertical",
			range: true,
			max: module.total,
			min: 0,
			step: 1000,
			values: [ module.validation, module.test+module.validation ],
			slide: function( event, ui ) {
				var h1 = parseInt(ui.values[0]);
				var h2 = parseInt(ui.values[1]);
				module.validation = h1;
				module.test = h2-h1;
				module.train = module.total-h2;
				
				// TODO dont use ids here?
				$('#validation').text(module.validation);
				$('#train').text(module.train);
				$('#test').text(module.test);
			}
		}).find(".ui-slider-handle").remove();
		
		dialog.find(".submit").remove();
	
	} else if(module.category==="Trainer"){
		dialog = renderTemplate("dialog", {
			id : id,
			type: "train",
			title : "Train your network",
			submit: "Train",
			cancel: "Delete"
		}, $(document.body));
		
		
		// form options
		renderTemplate("form-train", {
				id : module.id,
				method : module.method,
				loss : module.loss,
				batch: module.batch,
				learningRate: module.learningRate,
				momentum: module.momentum,
				regularization: module.regularization,
				clean: module.clean
			},
			dialog.find('.form-items'));
		
		
		dialog.find(".submit").click(function(e){
		    $(this).text(function(i, text){
		    	if(text === "Train"){
		    		var id = $(this).closest(".modal").find(".module-id").val();
					
					var trainer = learning[id];
					trainer.method = $(this).closest(".modal").find("#method").val();
					trainer.loss = $(this).closest(".modal").find("#loss").val();
					trainer.batch = $(this).closest(".modal").find("#batch").val();
					trainer.learningRate = $(this).closest(".modal").find("#learningRate").val();
					trainer.momentum = $(this).closest(".modal").find("#momentum").val();
					trainer.regularization = $(this).closest(".modal").find("#regularization").val();
					trainer.clean = $(this).closest(".modal").find("#clean").is(':checked');

					learn(id);
		    		return "Stop";
		    	} else {
		    		$.post("/dianne/learner", {"action":"stop",
		    			"id": nn.id});
		    		return "Train";
		    	}
		    	
		         return text === "Train" ? "Stop" : "Train";
		    });
			
		});
	} else if(module.category==="Evaluator"){
		dialog = renderTemplate("dialog", {
			id : id,
			type: "evaluate",
			title : "Evaluate your network",
			submit: "Evaluate",
			cancel: "Delete"
		}, $(document.body));
				
		
		dialog.find(".submit").click(function(e){
			var id = $(this).closest(".modal").find(".module-id").val();

			evaluate(id);
		});
	}

	// delete module on cancel
	dialog.find(".cancel").click(function(e){
		// remove object
		var id = $(this).closest(".modal").find(".module-id").val();
		
		var moduleItem = $('#'+id);
		if(checkRemoveModule(moduleItem)) {
			removeModule(moduleItem);
		}
		
		// remove dialog when module is removed, else keep it for reuse
		$(this).closest(".modal").remove();
	});
	
	return dialog;
}


/**
 * create dialogs for run modules
 */
function createRunModuleDialog(id, moduleItem){
	var module = running[id];
	if(module===undefined){
		return undefined; // no dialogs for build modules
	}
	
	var dialog;
	if(module.type==="CanvasInput"){
		dialog = renderTemplate("dialog", {
			id : id,
			type: "canvas",
			title : "Draw your input",
			submit: "",
			cancel: "Delete"
		}, $(document.body));
		
		dialog.find(".content").append("<canvas class='inputCanvas' width='224' height='224' style=\"border:1px solid #000000; margin-left:150px\"></canvas>");
		dialog.find(".content").append("<button class='btn' onclick='clearCanvas()' style=\"margin-left:10px\">Clear</button>");
		
		inputCanvas = dialog.find('.inputCanvas')[0];
		inputCanvasCtx = inputCanvas.getContext('2d');

		inputCanvasCtx.lineWidth = 15;
		inputCanvasCtx.lineCap = 'round';
		inputCanvasCtx.lineJoin = 'round';
		
		inputCanvas.addEventListener('mousemove', moveListener, false);
		inputCanvas.addEventListener('touchmove', touchMoveListener, false);
		inputCanvas.addEventListener('mousedown', downListener, false);
		inputCanvas.addEventListener('touchstart', downListener, false);
		inputCanvas.addEventListener('mouseup', upListener, false);
		inputCanvas.addEventListener('touchend', upListener, false);
		
		
	} else if(module.type==="RawInput"){
		dialog = renderTemplate("dialog", {
			id : id,
			type: "canvas",
			title : "Provide raw input data",
			submit: "",
			cancel: "Delete"
		}, $(document.body));
		
		dialog.find(".content").append("<textarea class=\"rawinput\" rows=\"5\" cols=\"65\">{\"dims\":[x,y,z],\"data\":[]}</textarea><br/><br/>");
		
		dialog.find(".content").append("<button class='btn' onclick='forwardRawInput(this, \""+module.input+"\")' style=\"margin-left:10px\">Submit</button>");
		
	} else if(module.category ==="Visualize" || module.type==="Youbot"){
		if(module.type ==="ProbabilityOutput"){		
			dialog = renderTemplate("dialog", {
				id : id,
				type: "probability",
				title : "Output probabilities",
				submit: "",
				cancel: "Delete"
			}, $(document.body));
			
			createOutputChart(dialog.find(".content"));
			var index = Number($("#dialog-"+module.id).find(".content").attr("data-highcharts-chart"));
			Highcharts.charts[index].yAxis[0].setExtremes(0,1);
			
		} else if(module.type ==="Youbot"){		
			dialog = renderTemplate("dialog", {
				id : id,
				type: "output",
				title : "Output values",
				submit: "",
				cancel: "Delete"
			}, $(document.body));
			
			createOutputChart(dialog.find(".content"));
			
			window.addEventListener("keydown", keyboard, false);
			window.addEventListener("keyup", keyboard, false);
		} else if(module.type === "LaserScan"){
			dialog = renderTemplate("dialog", {
				id : id,
				type: "laser",
				title : "LaserScan Output",
				submit: "",
				cancel: "Delete"
			}, $(document.body));
			
			dialog.find(".content").append("<canvas class='laserCanvas' width='512' height='512' style=\"border:1px solid #000000; margin-left:25px\"></canvas>");
		} else if(module.type === "TimeSeries"){
			dialog = renderTemplate("dialog", {
				id : id,
				type: "timeseries",
				title : "Timeseries output",
				submit: "",
				cancel: "Delete"
			}, $(document.body));
			createLineChart(dialog.find(".content"), '', 'Q-Value', []);
		} else {
			dialog = renderTemplate("dialog", {
				id : id,
				type: "outputviz",
				title : "Output",
				submit: "",
				cancel: "Delete"
			}, $(document.body));
		}

		dialog.find(".content").append("<div class='outputviz'></div>");
		dialog.find(".content").append("<div class='time'></div>");

		var eventsource = new EventSource("/dianne/run?nnId="+nn.id+"&moduleId="+module.output);
		eventsource.onmessage = function(event){
			var output = JSON.parse(event.data);
			if(output.error!==undefined){
				error(output.error);
			} else {
				$.each(running, function(id, module){
					// choose right RunOutput to set the chart of
					if(module.output===output.id){
						if(module.type ==="ProbabilityOutput"){
							var attr = $("#dialog-"+module.id).find(".content").attr("data-highcharts-chart");
							if(attr!==undefined){
								var index = Number(attr);
	
								if(output.tags.length != 0){
									var title = "";
									for(var i =0; i<output.tags.length; i++){
										if(!isFinite(String(output.tags[i]))){
											title+= output.tags[i]+" ";
										}
									}
									Highcharts.charts[index].setTitle({text: title});
								}
								Highcharts.charts[index].series[0].setData(output.probabilities, true, true, true);
								Highcharts.charts[index].xAxis[0].setCategories(output.labels);
							}
							$("#dialog-"+module.id).find(".content").find('.outputviz').hide();
						} else if(module.type === "Youbot"){
							var attr = $("#dialog-"+module.id).find(".content").attr("data-highcharts-chart");
							if(attr!==undefined){
								var index = Number(attr);
								if(output.data.length == 3){
									Highcharts.charts[index].series[0].setData(output.data, true, true, true);
									Highcharts.charts[index].xAxis[0].setCategories(['vx','vy','va']);
								} else if(output.data.length == 6){
									var data = output.data.splice(0, 3);
									var stdev = output.data;
									var errors = [];
									for (var i = 0; i < 3; i++) {
									   var tuple = [data[i]-stdev[i], data[i]+stdev[i]];
									   errors.push(tuple);
									}
									Highcharts.charts[index].series[0].setData(data, true, true, true);
									Highcharts.charts[index].series[1].setData(errors, true, true, true);
									Highcharts.charts[index].xAxis[0].setCategories(['vx','vy','va']);
								} else if(output.data.length == 7){
									if(output.probabilities===undefined){
										// DQN network, show raw Q values
										Highcharts.charts[index].series[0].setData(output.data, true, true, true);
										Highcharts.charts[index].xAxis[0].setCategories(['Left','Right','Forward','Backward','Turn Left','Turn Right', 'Grip']);
									} else {
										// policy network ending with softmax - show the probabilities
										Highcharts.charts[index].series[0].setData(output.probabilities, true, true, true);
										Highcharts.charts[index].xAxis[0].setCategories(['Left','Right','Forward','Backward','Turn Left','Turn Right','Grip']);
									}
								}
							}
						} else if(module.type === "LaserScan"){
							var laserCanvas = dialog.find('.laserCanvas')[0];
							var laserCanvasCtx = laserCanvas.getContext('2d');
							laser(output, laserCanvasCtx, false);
						} else if (module.type === "TimeSeries") {
							var attr = $("#dialog-"+module.id).find(".content").attr("data-highcharts-chart");
							if(attr!==undefined){
								var chart = Highcharts.charts[Number(attr)];
								while (chart.series.length < output.data.length) {
									chart.addSeries({data: []});
								}
								for (var i=0; i<output.data.length; i++) {
									var serie = chart.series[i];
									// shift if the series is longer than 100, higher numbers slow down the browser.
									var shift = serie.data.length > 100;
									// disable animation because adding points can be to fast to complete the animation
									serie.addPoint(output.data[i], true, shift, false);
								}
							}
						} else {
							// render raw output
							if(output.dims.length > 1){
								// as image
								var outputCanvas = dialog.find('.outputCanvas')[0];
								if(outputCanvas === undefined){
									$("#dialog-"+module.id).find(".content").append("<canvas class='outputCanvas' width='256' height='256' style=\"border:1px solid #000000; margin-left:150px\"></canvas>");
									 outputCanvas = dialog.find('.outputCanvas')[0];
								}
								var outputCanvasCtx = outputCanvas.getContext('2d');
								image(output, outputCanvasCtx);
							} else {
								// as floats
								$("#dialog-"+module.id).find(".content").find('.outputviz').html('<b>Output: </b>'+JSON.stringify(output.data).replace(/,/g,"  "));
								$("#dialog-"+module.id).find(".content").find('.outputviz').show();
							}
						}
						
						if(output.time === undefined){
							$("#dialog-"+module.id).find(".content").find('.time').hide();
						} else {
							$("#dialog-"+module.id).find(".content").find('.time').html('<b>Forward time: </b>'+output.time+' ms');
							$("#dialog-"+module.id).find(".content").find('.time').show();
						}
						
					}
				});
			}
		};
		
		dialog.on('hidden.bs.modal', function () {
			if($(".probability").length == 1){
				eventsource.close();
		    	eventsource = undefined;
			}
		    $(this).closest(".modal").remove();
		    
		    window.removeEventListener("keydown", keyboard, false);
			window.removeEventListener("keyup", keyboard, false);
		});
		
	} else if(module.category==="Dataset"){
		dialog = renderTemplate("dialog", {
			id : id,
			type: "dataset",
			title : "Input a sample of the "+module.type+" dataset",
			submit: "",
			cancel: "Delete"
		}, $(document.body));
		
		dialog.find(".content").append("<canvas class='sampleCanvas' width='256' height='256' style=\"border:1px solid #000000; margin-left:150px\"></canvas>");
		dialog.find(".content").append("<center><div class='expected'></div></center>");
		dialog.find(".content").append("<button class='btn' onclick='sample(\""+module.type+"\",\""+module.input+"\",this)' style=\"margin-left:10px\">Sample</button>");
		
		dialog.on('hidden.bs.modal', function () {
		    $(this).closest(".modal").remove();
		});
	} else if(module.type==="Camera"){
		dialog = renderTemplate("dialog", {
			id : id,
			type: "camera",
			title : "Camera input from "+module.name,
			submit: "",
			cancel: "Delete"
		}, $(document.body));
		
		dialog.find(".content").append("<canvas class='cameraCanvas' width='256' height='256' style=\"border:1px solid #000000; margin-left:150px\"></canvas>");

		var cameraCanvas = dialog.find('.cameraCanvas')[0];
		var cameraCanvasCtx = cameraCanvas.getContext('2d');
		
			var inputEventSource = new EventSource("/dianne/input?name=" + encodeURIComponent(module.name));
			inputEventSource.onmessage = function(event){
				var data = JSON.parse(event.data);
				image(data, cameraCanvasCtx);
			};
		
		dialog.on('hidden.bs.modal', function () {
		    inputEventSource.close();
		    inputEventSource = undefined;
		    $(this).closest(".modal").remove();
		});
	} else if(module.type==="LaserScanner"){ 
		dialog = renderTemplate("dialog", {
			id : id,
			type: "laserscanner",
			title : "LaserScan input from "+module.name,
			submit: "",
			cancel: "Delete"
		}, $(document.body));
		
		dialog.find(".content").append("<canvas class='laserCanvas' width='512' height='512' style=\"border:1px solid #000000; margin-left:25px\"></canvas>");
		dialog.find(".content").append("<br/><input type='checkbox' checked onclick='toggleTarget()'> show target position</input>");
		
		var laserCanvas = dialog.find('.laserCanvas')[0];
		var laserCanvasCtx = laserCanvas.getContext('2d');
		
		var inputEventSource = new EventSource("/dianne/input?name=" + encodeURIComponent(module.name));
		inputEventSource.onmessage = function(event){
			var tensor = JSON.parse(event.data);
			laser(tensor, laserCanvasCtx, laserTarget);
		};
		
		dialog.on('hidden.bs.modal', function () {
		    inputEventSource.close();
		    inputEventSource = undefined;
		    $(this).closest(".modal").remove();
		});
	} else if(module.type==="URLInput"){
		dialog = renderTemplate("dialog", {
			id : id,
			type: "url",
			title : "Give an URL to forward",
			submit: "",
			cancel: "Delete"
		}, $(document.body));
		
		dialog.find(".content").append("<img class='inputImage' width='224' height='auto' style=\"margin-left:150px\"></img><br/><br/>");
		dialog.find(".content").append("<input class='urlInput' size='50' value='http://'></input>");
		dialog.find(".content").append("<button class='btn' onclick='forwardURL(this, \""+module.input+"\")' style=\"margin-left:10px\">Forward</button>");
	} else if(module.type==="CharRNN"){
		dialog = renderTemplate("dialog", {
			id : id,
			type: "url",
			title : "Generate character sequences",
			submit: "",
			cancel: "Delete"
		}, $(document.body));
		
		dialog.find(".content").append("<div contenteditable=true class='charsequence'></div><br/><br/>");
		dialog.find(".content").append("Size: <input class='size' size='10' value='100'></input>");
		dialog.find(".content").append("<button class='btn' onclick='charsequence(this)' style=\"margin-left:10px\">Generate</button>");
	} else {
		dialog = createNNModuleDialog(module, "Configure run module", "", "Delete");
	}
	
	dialog.find(".cancel").click(function(e){
		// remove object
		var id = $(this).closest(".modal").find(".module-id").val();
		
		var moduleItem = $('#'+id);
		if(checkRemoveModule(moduleItem)) {
			removeModule(moduleItem);
		}
		
		// remove dialog when module is removed, else keep it for reuse
		$(this).closest(".modal").remove();
	});
	
	// submit button not used atm
	dialog.find(".submit").remove();
	
	return dialog;
}


/**
 * create dialogs for intermediate output
 */
function showConnectionDialog(connection) {
	var id = connection.sourceId
	var dialogId = "dialog-" + id;
	var dialog;
	dialog = $("#" + dialogId);
	if (dialog.length == 0) {
		var dialog;
		if (currentMode === "run") {
			dialog = createIntermediateOutputDialog(connection);
		}
		if (dialog !== undefined) {
			var offset = $("#"+id).offset();
			offset.top = offset.top - 100;
			offset.left = offset.left - 200;
			// show the modal (disable backdrop)
			dialog.draggable({
				handle : ".modal-header"
			}).mousedown(function(){
	   			// set clicked element to a higher level
	   			$(this).css('z-index', ++dialogZIndex);
			}).offset(offset);
		}
	}
	
	if (dialog !== undefined) {
		dialog.modal({
			'show' : true,
			'backdrop' : false
		}).css('z-index', ++dialogZIndex);
	}
}

function createIntermediateOutputDialog(connection){
	var dialog = renderTemplate("dialog", {
		id : connection.sourceId,
		type: "outputviz",
		title : "Intermediate output after "+nn.modules[connection.sourceId].name,
		submit: "",
		cancel: ""
	}, $(document.body));
	
	dialog.find(".content").append("<canvas class='outputCanvas' width='512' height='512' style=\"border:1px solid #000000; margin-left:25px\"></canvas>");
	dialog.find(".content").append("<div class='stats'>Min: <div class='min stat'></div> Mean: <div class='mean stat'></div> Max: <div class='max stat'></div> <div class='val stat'></div></div>");
	dialog.find('.outputCanvas').mousemove(function(e) {
	    var pos = findPos(this);
	    var x = e.pageX - pos.x;
	    var y = e.pageY - pos.y;
	    var coord = "x=" + x + ", y=" + y;
	    var c = this.getContext('2d');
	    var p = c.getImageData(x, y, 1, 1).data; 
	  
	    var min = parseFloat(dialog.find('.min').text());
	    var max = parseFloat(dialog.find('.max').text());
	    var val = (p[3]/255*(max-min)+min);
	    dialog.find('.val').text(isNaN(val) ? "" : val.toFixed(4));
	});
	
	var eventsource = new EventSource("/dianne/run?nnId="+nn.id+"&moduleId="+connection.sourceId);
	eventsource.onmessage = function(event){
		var output = JSON.parse(event.data);
		if(output.error===undefined){
			var dialog = $("#dialog-"+output.id);
			if(dialog !== undefined){
				var outputCanvas = dialog.find('.outputCanvas')[0];
				var outputCanvasCtx = outputCanvas.getContext('2d');
				image(output, outputCanvasCtx);
				dialog.find('.min').text(output.min.toFixed(4));
				dialog.find('.max').text(output.max.toFixed(4));
				dialog.find('.mean').text(output.mean.toFixed(4));
			}
		}
	};
	
	dialog.on('hidden.bs.modal', function () {
		if($(".probability").length == 1){
			eventsource.close();
	    	eventsource = undefined;
		}
	    $(this).closest(".modal").remove();
	});
		
	// submit button not used atm
	dialog.find(".submit").remove();
	dialog.find(".cancel").remove();
	
	return dialog;
}

function findPos(obj) {
    var curleft = 0, curtop = 0;
    if (obj.offsetParent) {
        do {
            curleft += obj.offsetLeft;
            curtop += obj.offsetTop;
        } while (obj = obj.offsetParent);
        return { x: curleft, y: curtop };
    }
    return undefined;
}

var inputCanvas;
var inputCanvasCtx;
var mousePos = {x: 0, y:0};

var laserTarget = true;

function toggleTarget(){
	laserTarget = !laserTarget;
}

function downListener(e) {
	e.preventDefault();
	inputCanvasCtx.moveTo(mousePos.x, mousePos.y);
	inputCanvasCtx.beginPath();
	inputCanvas.addEventListener('mousemove', onPaint, false);
	inputCanvas.addEventListener('touchmove', onPaint, false);
}

function upListener(e) {
	inputCanvas.removeEventListener('mousemove', onPaint, false);
	inputCanvas.removeEventListener('touchmove', onPaint, false);
	// get input 
	var canvasInputId = $(e.target.closest('.modal-body')).find('.module-id').val();
	forwardCanvasInput(running[canvasInputId].input);
}

function moveListener(e) {
	var dialog = inputCanvas.closest(".modal");
	mousePos.x = e.pageX - inputCanvas.offsetLeft - dialog.offsetLeft;
	mousePos.y = e.pageY - inputCanvas.offsetTop - dialog.offsetTop - 75;
}

function touchMoveListener(e) {
	var touches = e.targetTouches;
	var dialog = inputCanvas.closest(".modal");
	mousePos.x = touches[0].pageX - inputCanvas.offsetLeft - dialog.offsetLeft;
	mousePos.y = touches[0].pageY - inputCanvas.offsetTop - dialog.offsetTop - 75;
}

function onPaint() {
	// paint to big canvas
	inputCanvasCtx.lineTo(mousePos.x, mousePos.y);
	inputCanvasCtx.stroke();
}

function clearCanvas() {
	inputCanvasCtx.clearRect(0, 0, 224, 224);
}

function forwardCanvasInput(input){
	var array = [];
	var imageData = inputCanvasCtx.getImageData(0, 0, 224, 224);
    var data = imageData.data;
    
    // TODO hard coded for MNIST right now
    var sample = {};
    sample.dims = [1, 28, 28];
	for (var y = 0; y < 224; y+=8) {
        for (var x = 0; x < 224; x+=8) {
        	// collect alpha values
        	array.push(imageData.data[y*224*4+x*4+3]/255);
        }
    }
	sample.data = array;
	
	$.post("/dianne/run", {"forward":JSON.stringify(sample), "input":input, "id":nn.id}, 
			function( data ) {
			}
			, "json");
}

function forwardRawInput(btn, input){
	var tensor = $(btn).closest(".modal").find(".rawinput").val();

	$.post("/dianne/run", {"forward":tensor, "input":input, "id":nn.id}, 
			function( data ) {
			}
			, "json");
}

function forwardURL(btn, input){
	var url = $(btn).closest(".modal").find(".urlInput").val();
	
	$(btn).closest(".modal").find(".inputImage").attr("src", url);
	
	$.post("/dianne/run", {"url":url, "input":input, "id":nn.id}, 
			function( data ) {
			}
			, "json");
}

function sample(dataset, input, source){
	var uri = "/dianne/run";
	var args = {"dataset":dataset,"input":input, "id": nn.id};
	if(input==="undefined"){
		uri = "/dianne/datasets";
		args =  {"dataset":dataset,"action":"sample"};
	}
	$.post(uri, args , 
			function( sample ) {
				var sampleCanvas = $(source).parent().find('.sampleCanvas')[0];
				var sampleCanvasCtx = sampleCanvas.getContext('2d');
				render(sample, sampleCanvasCtx, datasets[dataset].inputType, datasets[dataset].labels);
				if(sample.target !== undefined)
					$('.expected').text('Expected output: '+sample.target);
			}
			, "json");
}

function charsequence(btn){
	var size = $(btn).closest(".modal").find(".size").val();
	var charsequence = $(btn).closest(".modal").find(".charsequence").text();
	
	var i = 0;
	var eventsource = new EventSource("/dianne/charrnn?id="+nn.id+"&size="+size+"&charsequence="+charsequence);
	eventsource.onmessage = function(event){
		var char = event.data;
		var textArea = $(btn).closest(".modal").find(".charsequence");
		textArea.html(textArea.html()+char);
		
		$('.charsequence').scrollTop($('.charsequence')[0].scrollHeight);
		
		if(size > 1000)
			i = i+100;
		else 
			i = i+1;
		
		if(i >= size-1)
			eventsource.close();
	}
}


/*
 * Deploy the modules
 */

function deployAll(){
	$("#spinnerwrap").show();
	$.post("/dianne/deployer", {"action":"deploy",
			"id":nn.id,
			"name":nn.name,
			"modules":JSON.stringify(nn.modules),
			"target":selectedTarget,
			"tags": $("#tags").val()}, 
			function( data ) {
				if(data.error!==undefined){
					error(data.error);
				} else {
					nn.id = data.id;
					$.each( data.deployment, color);
				}
				$("#spinnerwrap").hide();
			}
			, "json");
}

function undeployAll(){
	$.each(deployment, function(id,value){
		undeploy(id);
	});
}

function deploy(id, target){
	$("#spinnerwrap").show();
	$.post("/dianne/deployer", {"action":"deploy", 
		"id": nn.id,
		"name":nn.name,
		"module":JSON.stringify(nn.modules[id]),
		"target": target,
		"tags": $("#tags").val()}, 
			function( data ) {
				if(data.error!==undefined){
					error(data.error);
				} else {
					nn.id = data.id;
					$.each( data.deployment, color );
				}
				$("#spinnerwrap").hide();
			}
			, "json");
}

function undeploy(id){		
	$("#spinnerwrap").show();
	$.post("/dianne/deployer", {"action":"undeploy","id":nn.id,"moduleId":id}, 
			function( data ) {
				deployment[id] = undefined;
				$("#"+id).css('background-color', '');
				$("#spinnerwrap").hide();
			}
			, "json");
}

function color(id, target){
	deployment[id] = target;
	var c = deploymentColors[target]; 
	if(c === undefined){
		c = nextColor();
		deploymentColors[target] = c;
	}
	$("#"+id).css('background-color', c);
}

/*
 * Learning functions
 */

function learn(id){
	// first create the chart
	createLossChart($("#dialog-"+id).find(".content"));

	var eventsource = new EventSource("/dianne/learner?nnId="+nn.id);
	eventsource.onmessage = function(event){
		var data = JSON.parse(event.data);
		
		if(data.sample===undefined){
			error(data.error);
		} else {
			var index = Number($("#dialog-"+id).find(".content").attr("data-highcharts-chart"));
    		var x = Number(data.sample);
        	var y = Number(data.loss); 
			Highcharts.charts[index].series[0].addPoint([x, y], true, true, false);
		}
	};
	
	$.post("/dianne/learner", {"action":"learn",
		"id": nn.id,
		"config":JSON.stringify(learning),
		"target": id}, 
			function( data ) {
				// only returns labels of output module
				$.each(data, function(id, labels){
					nn.modules[id].labels = labels;
				});
				eventsource.close();
				eventsource = undefined;
			}
			, "json");
}

function evaluate(id){
	$("#dialog-"+id).find(".content").empty();
	
	$("#spinnerwrap").show();
	$.post("/dianne/learner", {"action":"evaluate",
		"id": nn.id,
		"config":JSON.stringify(learning),
		"target": id}, 
			function( data ) {
				if(data.confusionMatrix!==undefined){
					// confusion chart and accuracy div
					createConfusionChart($("#dialog-"+id).find(".content"));
					$("#dialog-"+id).find(".content").append("<div>Accuracy: "+data.accuracy+" %</div>");
					var index = Number($("#dialog-"+id).find(".content").attr("data-highcharts-chart"));
					Highcharts.charts[index].series[0].setData(data.confusionMatrix, true, true, false);
				} else {
					$("#dialog-"+id).find(".content").append("<div>Error: "+data.error+"</div>");
				} 
				$("#spinnerwrap").hide();
			}
			, "json");
}

// code block so global variables are not defined
{
/*
 * SSE for feedback when training/running
 */
var eventsource;

if(typeof(EventSource) === "undefined") {
	// load polyfill eventsource library
	$.getScript( "js/lib/eventsource.min.js").done(function( script, textStatus ) {
		console("Fallback to eventsource.js for SSE...");
	}).fail(function( jqxhr, settings, exception ) {
		console.log("Sorry, your browser does not support server-sent events...");
	});
} 
}



/*
 * Charts
 */

function createOutputChart(container) {
    return container.highcharts({
        chart: {
            type: 'column',
    		height: 300,
    		width: 500
        },
        title: {
            text: null
        },
        xAxis: {
            type: 'category',
            labels: {
                rotation: -45
            }
        },
        yAxis: {
            title: {
                text: null
            }
        },
        legend: {
            enabled: false
        },
        series: [{
            name: 'Output',
            type: 'column'
        },
        {
        	name: 'Error',
        	type: 'errorbar'
        }]
    });
}


function createLossChart(container) {
    return container.highcharts({
        chart: {
            type: 'line',
            animation: false, // don't animate in old IE
            marginRight: 10,
    		height: 200,
    		width: 500
        },
        title : {
        	text: null
        },
        xAxis: {
            tickPixelInterval: 150
        },
        yAxis: {
            title: {
                text: 'Loss'
            },
            min: 0,
            plotLines: [{
                value: 0,
                width: 1,
                color: '#808080'
            }]
        },
        legend: {
            enabled: false
        },
        exporting: {
            enabled: false
        },
        series: [{
            name: 'Data',
            data: (function () {
                // generate an array of empty data
                var data = [],i;
                for (i = -29; i <= 0; i += 1) {
                    data.push({
                        x: 0,
                        y: null
                    });
                }
                
                return data;
            }())
        }]
    });
}

function createConfusionChart(container) {
    return container.highcharts({
    	chart: {
            type: 'heatmap',
    		height: 500,
    		width: 500
        },
        title: {
            text: "Confusion Matrix"
        },
        colorAxis: {
            stops: [
                [0, '#3060cf'],
                [0.5, '#fffbbc'],
                [0.9, '#c4463a']
            ],
            min: 0
//            min: 0,
//            minColor: Highcharts.getOptions().colors[0],
//            maxColor: '#FFFFFF'
        },
        yAxis: {
            title: {
                text: null
            }
        },
        series: [{
            name: 'Confusion matrix',
            borderWidth: 0,
            dataLabels: {
                enabled: false,
                color: 'black',
                style: {
                    textShadow: 'none',
                    HcTextStroke: null
                }
            }
        }]
    });
}

//generic line chart
function createLineChart(container, xAxis, yAxis, series) {	
    return container.highcharts({
    	chart: {
            type: 'line',
            animation: false, // don't animate in old IE
            marginRight: 10,
    		height: 200,
    		width: 500, 
    		zoomType: 'x'
        },
        title : {
        	text: null
        },
        xAxis: {
            tickPixelInterval: 150,
            title: {
                text: xAxis
            },
        },
        plotOptions:{
            series:{
                turboThreshold: 1000000
            }
        },
        yAxis: {
            title: {
                text: yAxis
            },
            plotLines: [{
                value: 0,
                width: 1,
                color: '#808080'
            }],
            softMin: 0,
            softMax: 1
        },
        credits: {
            enabled: false
        },
        series: series
    });
}

// error handler
function error(message){
	renderTemplate("error", {
		'message' : message
	}, $("#alerts"));
}

// keyboard control handler
function keyboard(e) {
	$.post("/keyboard/servlet",{'type':e.type,'key':e.key,'code':e.code})
}