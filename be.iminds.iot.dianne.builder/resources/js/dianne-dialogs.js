/*
 * Module configuration/deletion dialog stuff 
 */

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
	}
	
	if (dialog !== undefined) {
		var offset = moduleItem.offset();
		offset.top = offset.top - 100;
		offset.left = offset.left - 200;
		// show the modal (disable backdrop)
		dialog.modal({
			'show' : true,
			'backdrop' : false
		}).draggable({
			handle : ".modal-header"
		}).offset(offset);
	}
}

/**
 * Create dialog for configuring module in build mode
 */
function createBuildModuleDialog(id, moduleItem){
	var module = nn[id];
	
	var dialog = renderTemplate("dialog", {
		id : id,
		title : "Configure module ",
		submit: "Configure",
		cancel: "Delete"
	}, $(document.body));
	
	// add module div to dialog to show which module to configure
	renderTemplate("module",
			{	
				name: module.type,
				type: module.type, 
				category: module.category
			}, 
			dialog.find('.content'));
	
	// then fill in properties
	$.post("/dianne/builder", {"action" : "module-properties","type" : module.type}, 
			function( data ) {
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
					module = nn[item.value];
				} else {
					module[item.name] = item.value;
				}
			});
			
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
	var module = nn[id];
	
	var dialog = renderTemplate("dialog", {
		id : id,
		title : "Deploy module ",
		submit: "Deploy",
		cancel: "Undeploy"
	}, $(document.body));
	
	// add module div to dialog to show which module to configure
	renderTemplate("module",
			{	
				name: module.type,
				type: module.type, 
				category: module.category
			}, 
			dialog.find('.content'));
	
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
						dialog.find('.options').append("<option value="+target+">"+target+"</option>")
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
		return undefined; // no dialogs for build modules
	}
	
	var dialog;
	if(module.category==="Dataset"){
		dialog = renderTemplate("dialog", {
			id : id,
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
		
		// TODO make this a shuffle button?
		dialog.find(".submit").remove();
	
	} else if(module.category==="Trainer"){
		dialog = renderTemplate("dialog", {
			id : id,
			title : "Train your network",
			submit: "Train",
			cancel: "Delete"
		}, $(document.body));
		
		
		// form options
		// TODO fetch parameters from server?
		renderTemplate("form-train", {
				id : module.id,
				loss : module.loss,
				batch: module.batch,
				epochs: module.epochs,
				learningRate: module.learningRate
			},
			dialog.find('.form-items'));
		
		
		dialog.find(".submit").click(function(e){
			var id = $(this).closest(".modal").find(".module-id").val();
			
			var trainer = learning[id];
			trainer.loss = $(this).closest(".modal").find("#loss").val();
			trainer.batch = $(this).closest(".modal").find("#batch").val();
			trainer.epochs = $(this).closest(".modal").find("#epochs").val();
			trainer.learningRate = $(this).closest(".modal").find("#learningRate").val();

			learn(id);
		});
	} else if(module.category==="Evaluator"){
		dialog = renderTemplate("dialog", {
			id : id,
			title : "Evaluate your network",
			submit: "Evaluate",
			cancel: "Delete"
		}, $(document.body));
				
		// confusion chart and accuracy div
		createConfusionChart(dialog.find(".content"));
		dialog.find(".content").append("<div class=\"accuracy\"></div>")
		
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
		
		
	} else if(module.type==="ProbabilityOutput"){
		dialog = renderTemplate("dialog", {
			id : id,
			title : "Output probabilities",
			submit: "",
			cancel: "Delete"
		}, $(document.body));
		
		createOutputChart(dialog.find(".content"));
		eventsource = new EventSource("run");
		eventsource.onmessage = function(event){
			var data = JSON.parse(event.data);
			var index = Number($("#dialog-"+id).find(".content").attr("data-highcharts-chart"));
			Highcharts.charts[index].series[0].setData(data, true, true, true);
		};
		
		dialog.on('hidden.bs.modal', function () {
		    eventsource.close();
		});
		
	} else if(module.category==="Dataset"){
		dialog = renderTemplate("dialog", {
			id : id,
			title : "Input a sample of the "+module.type+" dataset",
			submit: "",
			cancel: "Delete"
		}, $(document.body));
		
		dialog.find(".content").append("<canvas class='sampleCanvas' width='224' height='224' style=\"border:1px solid #000000; margin-left:150px\"></canvas>");
		dialog.find(".content").append("<button class='btn' onclick='sample()' style=\"margin-left:10px\">Sample</button>");
		
		sampleCanvas = dialog.find('.sampleCanvas')[0];
		sampleCanvasCtx = sampleCanvas.getContext('2d');
		
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

var inputCanvas;
var inputCanvasCtx;
var mousePos = {x: 0, y:0};

var sampleCanvas;
var sampleCanvasCtx;

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
	forwardCanvasInput();
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

function forwardCanvasInput(){
	var array = [];
	var imageData = inputCanvasCtx.getImageData(0, 0, 224, 224);
    var data = imageData.data;

	for (var y = 0; y < 224; y+=8) {
        for (var x = 0; x < 224; x+=8) {
        	// collect alpha values
        	array.push(imageData.data[y*224*4+x*4+3]/255);
        }
    }
	$.post("/dianne/run", {"forward":JSON.stringify(array)}, 
			function( data ) {
			}
			, "json");
}

function sample(){
	$.post("/dianne/run", {"sample":"random"}, 
			function( data ) {
				var imageData = sampleCanvasCtx.createImageData(224, 224);
				for (var y = 0; y < 224; y++) {
			        for (var x = 0; x < 224; x++) {
			        	// collect alpha values
			        	var x_s = Math.floor(x/8);
			        	var y_s = Math.floor(y/8);
			        	var index = y_s*28+x_s;
			        	imageData.data[y*224*4+x*4+3] = Math.floor(data[index]*255);
			        }
			    }
				sampleCanvasCtx.putImageData(imageData, 0, 0); 
				
				$.post("/dianne/run", {"forward":JSON.stringify(data)}, 
						function( data ) {
						}
						, "json");
			}
			, "json");
	
}


/*
 * Deploy the modules
 */

function deployAll(){
	$.post("/dianne/deployer", {"action":"deploy","modules":JSON.stringify(nn)}, 
			function( data ) {
				$.each( data, function(id,target){
					deployment[id] = target;
					// TODO separate color per node?
					$("#"+id).css('background-color', '#FF6CDA');
				});
			}
			, "json");
}

function undeployAll(){
	$.each(deployment, function(id,value){
		undeploy(id);
	});
}

function deploy(id, target){
	$.post("/dianne/deployer", {"action":"deploy",
		"module":JSON.stringify(nn[id]),
		"target": target}, 
			function( data ) {
				$.each( data, function(id,target){
					deployment[id] = target;
					var c = deploymentColors[target]; 
					if(c === undefined){
						c = nextColor();
						deploymentColors[target] = c;
					}
					$("#"+id).css('background-color', c);
				});
			}
			, "json");
}

function undeploy(id){
	$.post("/dianne/deployer", {"action":"undeploy","id":id}, 
			function( data ) {
				deployment[id] = undefined;
				$("#"+id).css('background-color', '');
			}
			, "json");
}

var deploymentColors = {};
var colors = ['#FF6CDA','#81F781','#AC58FA','#FA5858'];
var colorIndex = 0;

function nextColor(){
	return colors[colorIndex++];
}

/*
 * Learning functions
 */

function learn(id){
	// first create the chart
	createErrorChart($("#dialog-"+id).find(".content"));

	eventsource = new EventSource("learner");
	eventsource.onmessage = function(event){
		var data = JSON.parse(event.data);
		var index = Number($("#dialog-"+id).find(".content").attr("data-highcharts-chart"));
    	var x = Number(data.sample);
        var y = Number(data.error); 
		Highcharts.charts[index].series[0].addPoint([x, y], true, true, false);
	};
	$.post("/dianne/learner", {"action":"learn",
		"config":JSON.stringify(learning),
		"target": id}, 
			function( data ) {
				$.each(data, function(id, parameters){
					nn[id].parameters = parameters;
				});
				eventsource.close();
			}
			, "json");
}

function evaluate(id){
	// reset chart
	var index = Number($("#dialog-"+id).find(".content").attr("data-highcharts-chart"));
	Highcharts.charts[index].series[0].setData(null, true, true, false);
	$("#dialog-"+id).find(".accuracy").text("");

	eventsource = new EventSource("learner");
	eventsource.onmessage = function(event){
		var data = JSON.parse(event.data);
		var index = Number($("#dialog-"+id).find(".content").attr("data-highcharts-chart"));
		Highcharts.charts[index].series[0].setData(data, true, true, false);
	};
	$.post("/dianne/learner", {"action":"evaluate",
		"config":JSON.stringify(learning),
		"target": id}, 
			function( data ) {
				eventsource.close();
				$("#dialog-"+id).find(".accuracy").text("Accuracy: "+data.accuracy+" %");
			}
			, "json");
}

/*
 * SSE for feedback when training/running
 */
var eventsource;

if(typeof(EventSource) === "undefined") {
	// load polyfill eventsource library
	$.getScript( "js/eventsource.min.js").done(function( script, textStatus ) {
		console("Fallback to eventsource.js for SSE...");
	}).fail(function( jqxhr, settings, exception ) {
		console.log("Sorry, your browser does not support server-sent events...");
	});
} 



/*
 * Charts
 */

function createOutputChart(container) {
    container.highcharts({
        chart: {
            type: 'column',
    		height: 300,
    		width: 500
        },
        title: {
            text: null
        },
        xAxis: {
            type: 'category'
        },
        yAxis: {
            min: 0,
            max: 1,
            title: {
                text: null
            }
        },
        legend: {
            enabled: false
        },
        series: [{
            name: 'Output'
        }]
    });
}


function createErrorChart(container) {
    container.highcharts({
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
                text: 'Error'
            },
            max: 1,
            floor: 0,
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
                for (i = -19; i <= 0; i += 1) {
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
    container.highcharts({
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