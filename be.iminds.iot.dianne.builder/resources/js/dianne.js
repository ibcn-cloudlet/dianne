/**
 * This script allows to create a NN structure by drag-and-drop using jsPlumb
 */

// keep a map of neural network modules
var modules = {};
// keep a map of all learning blocks
var learning = {};

// keep map module id -> deployment node
var deployment = {};

/*
 * UI Mode
 */
var modus = "build";

function setModus(m){
	$(".active").removeClass("active");
	// only show learn modules in learn modus
	$(".learn").each(function( index ) {
		jsPlumb.hide($(this).attr('id'),true);
		$(this).hide();
	});
	modus = m;
	if(modus === "build"){
		console.log("switch to build");
		$(".toolbox").hide();
		$("#menu-build").addClass("active");
		$("#toolbox-build").show();
		$("#toolbox-build").addClass("active");
	} else if(modus === "deploy"){
		console.log("switch to deploy");
		$(".toolbox").hide();
		$("#menu-deploy").addClass("active");
		$("#toolbox-deploy").show();
		$("#toolbox-deploy").addClass("active");
	} else if(modus === "learn"){
		console.log("switch to learn");
		$(".toolbox").hide();
		$("#menu-learn").addClass("active");
		$("#toolbox-learn").show();
		$("#toolbox-learn").addClass("active");
		// only show learn modules in learn modus
		$(".learn").each(function( index ) {
			jsPlumb.show($(this).attr('id'),true);
			$(this).show();
		});
	} else if(modus === "run"){
		console.log("switch to run");
		$(".toolbox").hide();
		$("#menu-run").addClass("active");
		$("#toolbox-run").show();
		$("#toolbox-run").addClass("active");
	}
	// hide all modals
	$(".modal").modal('hide');
}


/*
 * jsPlumb rendering and setup
 */

// definition of source Endpoints
var source = {
	isSource:true,
	anchor : "Right",	
	paintStyle:{ 
		strokeStyle:"#555", 
		fillStyle:"#FFF", 
		lineWidth:2 
	},
	hoverPaintStyle:{
		lineWidth:3 
	},			
	connectorStyle:{
		lineWidth:4,
		strokeStyle:"#333",
		joinstyle:"round",
		outlineColor:"white",
		outlineWidth:2
	},
	connectorHoverStyle:{
		lineWidth:4,
		strokeStyle:"#555",
		outlineWidth:2,
		outlineColor:"white"
	}
}		

// the definition of target Endpoints 
var target = {
	isTarget:true,
	anchor: "Left",					
	paintStyle:{ 
		fillStyle:"#333"
	},
	hoverPaintStyle:{ 
		fillStyle: "#555"
	}
}

/**
 * On ready, fill the toolbox with available supported modules
 */
$( document ).ready(function() {
	// initialize toolboxes
	// build toolbox
	$.post("/dianne/builder", {action : "available-modules"}, 
		function( data ) {
			$.each(data, function(index, name){
				addToolboxItem('#toolbox-build', name, name, 'build');
			});
		}
		, "json");
	
	// learn toolbox
	// TODO this is hard coded for now, as this does not map to factories/module impls
	addToolboxItem('#toolbox-learn','MNIST Dataset','Dataset','learn');
	addToolboxItem('#toolbox-learn','SGD Trainer','Trainer','learn');
	addToolboxItem('#toolbox-learn','Arg Max Evaluator','Evaluator','learn');
	
	// show correct mode
	setModus(modus);
});

// add a toolbox item name to toolbox with id toolboxId and add class clazz
function addToolboxItem(toolboxId, name, type, clazz){
	$(toolboxId).append(renderTemplate("module",
			{name: name, type: type }));
	
	// make draggable and add code to create new modules drag-and-drop style
	$('#'+type).draggable({helper: "clone"});
	$('#'+type).bind('dragstop', function(event, ui) {
		if(checkAddModule($(this))){
			// clone the toolbox item
		    var moduleItem = $(ui.helper).clone().removeClass("toolbox").addClass(clazz);
		    
			addModule(moduleItem, $(this));
		}
	});
}

// jsPlumb init code
jsPlumb.ready(function() {       
    jsPlumb.setContainer($("canvas"));
    jsPlumb.importDefaults({
    	ConnectionOverlays : [[ "Arrow", { location : 1 } ]],
    	Connector : [ "Flowchart", { stub:[40, 60], gap:10, cornerRadius:5, alwaysRespectStubs:true } ],
    	DragOptions : { cursor: 'pointer', zIndex:2000 },
    });		

	// suspend drawing and initialise.
	jsPlumb.doWhileSuspended(function() {
		//
		// listen for connection add/removes
		//
		jsPlumb.bind("beforeDrop", function(connection) {
			if(!checkAddConnection(connection)){
				return false;
			}
			addConnection(connection);
			return true;
		});
		
		jsPlumb.bind("beforeDetach", function(connection) {
			if(!checkRemoveConnection(connection)){
				return false;
			}
			removeConnection(connection);
			return true;
		});
	});

});



/*
 * Module/Connection add/remove checks
 */

/**
 * Check whether one is allowed to instantiate another item from this tooblox
 */
function checkAddModule(toolboxItem){
	return true;
}

/**
 * Check whether one is allowed to remove this module
 */
function checkRemoveModule(moduleItem){
	return true;
}

/**
 * Check whether one is allowed to instantiate this connection
 */
function checkAddConnection(connection){
	if(modus==="build"){
		if(deployment[connection.sourceId]!==undefined
				|| deployment[connection.targetId]!==undefined){
				return false;
		}
		if(connection.connection.endpoints[0].type!=="Dot" 
			|| connection.connection.endpoints[1].type!=="Dot"){
				return false;
		}
	}
	if(modus==="learn"){
		if(connection.connection.endpoints[0].type!=="Rectangle" 
			|| connection.connection.endpoints[1].type!=="Rectangle"){
				return false;
		}
		//TODO dont allow connecting output to input
	}
	return true;
}

/**
 * Check whether one is allowed to remove this connection
 */
function checkRemoveConnection(connection){
	if(deployment[connection.sourceId]!==undefined
		|| deployment[connection.targetId]!==undefined){
		return false;
	}
	if(modus==="build"){
		if(connection.connection.endpoints[0].type!=="Dot" 
			|| connection.connection.endpoints[1].type!=="Dot"){
				return false;
		}
	}
	if(modus==="learn"){
		if(connection.connection.endpoints[0].type!=="Rectangle" 
			|| connection.connection.endpoints[1].type!=="Rectangle"){
				return false;
		}
	}
	return true;
}


/*
 * Module/Connection add/remove methods
 */

/**
 * Add a module to the canvas and to modules datastructure
 * 
 * @param moduleItem a freshly cloned DOM element from toolbox item 
 * @param toolboxItem the toolbox DOM element the moduleItem was cloned from
 */
function addModule(moduleItem, toolboxItem){
	moduleItem.appendTo("#canvas");
	 
    // fix offset of toolbox 
    var offset = {};
    offset.left = moduleItem.offset().left - ($("#canvas").offset().left - $(".toolbox.active").offset().left);
    offset.top = moduleItem.offset().top - ($("#canvas").offset().top - $(".toolbox.active").offset().top);
    moduleItem.offset(offset);
  
    // get type from toolbox item and generate new UUID
	var type = toolboxItem.attr("id");
	var id = guid();
	moduleItem.attr("id",id);
	
	// TODO this should not be hard coded?
	if(type==="Input"){
		jsPlumb.addEndpoint(moduleItem, source);
		jsPlumb.addEndpoint(moduleItem, target, {endpoint:"Rectangle",filter:":not(.build)"});
	} else if(type==="Output"){
		jsPlumb.addEndpoint(moduleItem, source, {endpoint:"Rectangle", maxConnections:-1});
		jsPlumb.addEndpoint(moduleItem, target);
	} else if(type==="Trainer" || type==="Evaluator"){
		jsPlumb.addEndpoint(moduleItem, target, {endpoint:"Rectangle"});
	} else if(type==="Dataset"){ 
		jsPlumb.addEndpoint(moduleItem, source, {endpoint:"Rectangle"});
	} else {
		jsPlumb.addEndpoint(moduleItem, source);
		jsPlumb.addEndpoint(moduleItem, target);
	}
	
	// show dialog on double click
	moduleItem.dblclick(function() {
		showConfigureModuleDialog($(this));
	});
	
	// make draggable
	moduleItem.draggable(
	{
		drag: function(){
		    jsPlumb.repaintEverything();
		}
	});
	
	// add to modules
	var module = {};
	module.type = type;
	module.id = id;
	
	if(type!=="Trainer" 
		&& type!=="Evaluator"
		&& type!=="Dataset"){
		modules[id] = module;
	} else {
		if(type==="Dataset"){
			// TODO this is hard coded for MNIST
			module.dataset = "MNIST";
			module.total = 70000;
			module.train = 60000;
			module.test = 10000;
			module.validation = 0;
			
		} else if(type==="Trainer"){
			// TODO this is hard coded
			//module.strategy = "Stochastic Gradient Descent";
			module.batch = 10;
			module.epochs = 1;
			module.loss = "MSE";
		}
		learning[id] = module;
	}
	
	console.log("Add module "+id);
}

/**
 * Remove a module from the canvas and the modules datastructure
 * 
 * @param moduleItem the DOM element on the canvas representing the module
 */
function removeModule(moduleItem){
	var id = moduleItem.attr("id");

	// delete this moduleItem
	$.each(jsPlumb.getEndpoints(moduleItem), function(index, endpoint){
		jsPlumb.deleteEndpoint(endpoint)}
	);
	
	jsPlumb.detachAllConnections(moduleItem);
	moduleItem.remove();

	// remove from modules
	if(modules[id]!==undefined){
		if(modules[modules[id].next]!==undefined){
			delete modules[modules[id].next].prev;
		}
		if(modules[modules[id].prev]!==undefined){
			delete modules[modules[id].prev].next;
		}
		delete modules[id];
	} else {
		delete learning[id];
	}
	console.log("Remove module "+id);
	
}

/**
 * Add a connection between two modules
 * @param connection to add
 */
function addConnection(connection){
	console.log("Add connection " + connection.sourceId + " -> " + connection.targetId);
	if(learning[connection.sourceId]!==undefined){
		// dataset
		learning[connection.sourceId].input = connection.targetId; 
	} else if(learning[connection.targetId]!==undefined){
		// trainer/evaluator
		learning[connection.targetId].output = connection.sourceId; 
	} else {
		modules[connection.sourceId].next = connection.targetId;
		modules[connection.targetId].prev = connection.sourceId;
	}
}

/**
 * Remove a connection between two modules
 * @param connection to remove
 */
function removeConnection(connection){
	console.log("Remove connection " + connection.sourceId + " -> " + connection.targetId);
	if(learning[connection.sourceId]!==undefined){
		// dataset
		delete learning[connection.sourceId].input; 
	} else if(learning[connection.targetId]!==undefined){
		// trainer/evaluator
		delete learning[connection.targetId].output; 
	} else {
		delete modules[connection.sourceId].next;
		delete modules[connection.targetId].prev;
	}
}


/*
 * Module configuration/deletion dialog stuff 
 */

function showConfigureModuleDialog(moduleItem){
	var id = moduleItem.attr("id");
	
	// there can be only one dialog at a time for one module
	// try to reuse dialog
	var dialogId = "dialog-"+id;
	var dialog;
	dialog = $("#"+dialogId);
	if(dialog.length==0){
		// create new dialog
		var d = renderTemplate("dialog", {
			id : id,
			title : "Configure module "
		});
		dialog = $(d);
	}
	
	// TODO check which "mode" you are in, for now only "build" mode
	if(modus==="build"){
		dialog = createBuildModuleDialog(id, dialog);
	} else if(modus==="deploy"){
		dialog = createDeployModuleDialog(id, dialog);
	} else if(modus==="learn"){
		dialog = createLearnModuleDialog(id, dialog);
	}
	
	if(dialog!==undefined){
		var offset = moduleItem.offset();
		offset.top = offset.top - 100;
		offset.left = offset.left - 200;
	
		// show the modal (disable backdrop)
		dialog.modal({'show':true, 'backdrop':false}).draggable({handle: ".modal-header"}).offset(offset);
	}
}

function createBuildModuleDialog(id, dialog){
	var module = modules[id];
	
	// create build body form
	var body = renderTemplate("dialog-body-build", {
		id : module.id,
		type : module.type
	});
	dialog.find(".modal-body").empty();
	dialog.find(".modal-body").append(body);
	
	// then fill in properties
	$.post("/dianne/builder", {"action" : "module-properties","type" : module.type}, 
		function( data ) {
			$.each(data, function(index, property){
				// Render toolbox item
				dialog.find('.form-properties').append(
						renderTemplate("form-properties", 
						{
							name: property.name,
							id: property.id,
							value: module[property.id]
						}));
			});
			if (data.length === 0) {
				dialog.find('.form-properties').append("<p>No properties to configure...</p>");
			}
		}
		, "json");
	
	// add buttons
	var buttons = renderTemplate("dialog-buttons-build", {});
	dialog.find(".modal-footer").empty();
	dialog.find(".modal-footer").append(buttons);
	
	// add button callbacks, disable buttons when module is deployed
	if(deployment[id]!==undefined){
		dialog.find(".configure").prop('disabled', true);
		dialog.find(".delete").prop('disabled', true);
	} else {
		dialog.find(".configure").click(function(e){
			// apply configuration
			var data = $(this).closest('.modal').find('form').serializeArray();
			
			var module;
			$.each( data, function( i, item ) {
				if(i === 0){
					module = modules[item.value];
				} else {
					module[item.name] = item.value;
				}
			});
			
			$(this).closest(".modal").modal('hide');
		});
		
		dialog.find(".delete").click(function(e){
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


function createDeployModuleDialog(id, dialog){
	var module = modules[id];
	
	// create build body form
	var body = renderTemplate("dialog-body-deploy", {
		id : module.id,
		type : module.type
	});
	dialog.find(".modal-body").empty();
	dialog.find(".modal-body").append(body);
	
	// fill in deployment options
	if(deployment[id]===undefined){
		dialog.find('.form-deployment').append(
				renderTemplate("form-deployment", 
				{}));
		$.post("/dianne/deployer", {"action" : "targets"}, 
				function( data ) {
					$.each(data, function(index, target){
						dialog.find('.targets').append("<option value="+target+">"+target+"</option>")
					});
				}
				, "json");
	} else {
		dialog.find('.form-deployment').append("<p>This module is deployed to "+deployment[id]+"</p>");
	}
	
	// add buttons
	var buttons = renderTemplate("dialog-buttons-deploy", {});
	dialog.find(".modal-footer").empty();
	dialog.find(".modal-footer").append(buttons);
	
	// add button callbacks
	if(deployment[id]===undefined){
		dialog.find(".deploy").click(function(e){
			// deploy this module
			var id = $(this).closest(".modal").find(".module-id").val();
			var target = $(this).closest('.modal').find('.targets').val();
			
			deploy(id, target);
			
			$(this).closest(".modal").modal('hide');
		});
		dialog.find(".undeploy").prop('disabled', true);
	} else {
		dialog.find(".undeploy").click(function(e){
			// undeploy this module
			var id = $(this).closest(".modal").find(".module-id").val();
			undeploy(id);
			
			// remove dialog when module is removed, else keep it for reuse
			$(this).closest(".modal").modal('hide');
		});
		dialog.find(".deploy").prop('disabled', true);
	}
	
	
	return dialog;
}


function createLearnModuleDialog(id, dialog){
	var block = learning[id];
	if(block===undefined){
		return undefined; // (for now) only dialogs for learning blocks
	}
	
	var buttons = renderTemplate("dialog-buttons-learn", {});
	dialog.find(".modal-footer").empty();
	dialog.find(".modal-footer").append(buttons);
	
	dialog.find(".delete").click(function(e){
		// remove object
		var id = $(this).closest(".modal").find(".module-id").val();
		
		var moduleItem = $('#'+id);
		if(checkRemoveModule(moduleItem)) {
			removeModule(moduleItem);
		}
		
		// remove dialog when module is removed, else keep it for reuse
		$(this).closest(".modal").remove();
	});
	
	
	if(block.type==="Dataset"){
		var body = renderTemplate("dialog-body-dataset", {
			id : block.id,
			dataset : block.dataset,
			train: block.train,
			test: block.test,
			validation: block.validation
		});
		dialog.find(".modal-body").empty();
		dialog.find(".modal-body").append(body);
		dialog.find(".slider").slider({
			orientation: "vertical",
			range: true,
			max: block.total,
			min: 0,
			step: 1000,
			values: [ block.validation, block.test+block.validation ],
			slide: function( event, ui ) {
				var h1 = parseInt(ui.values[0]);
				var h2 = parseInt(ui.values[1]);
				block.validation = h1;
				block.test = h2-h1;
				block.train = block.total-h2;
				
				$('#validation').text(block.validation);
				$('#train').text(block.train);
				$('#test').text(block.test);
			}
		}).find(".ui-slider-handle").remove();
		
		// TODO make this a shuffle button?
		dialog.find(".run").remove();
		
		dialog.find(".modal-title").text("Configure dataset");
	} else if(block.type==="Trainer"){
		var body = renderTemplate("dialog-body-train", {
			id : block.id,
			loss : block.loss,
			batch: block.batch,
			epochs: block.epochs
		});
		dialog.find(".modal-body").empty();
		dialog.find(".modal-body").append(body);
		
		dialog.find(".modal-title").text("Train the network");
		
		dialog.find(".run").click(function(e){
			var id = $(this).closest(".modal").find(".module-id").val();
			
			var trainer = learning[id];
			trainer.loss = $(this).closest(".modal").find("#loss").val();
			trainer.batch = $(this).closest(".modal").find("#batch").val();
			trainer.epochs = $(this).closest(".modal").find("#epochs").val();

			learn(id);
		});
	} else if(block.type==="Evaluator"){
		var body = renderTemplate("dialog-body-evaluate", {id : block.id});
		dialog.find(".modal-body").empty();
		dialog.find(".modal-body").append(body);
		
		dialog.find(".modal-title").text("Evaluate the network");
				
		createConfusionChart(dialog.find(".evaluate"));
		
		dialog.find(".run").click(function(e){
			var id = $(this).closest(".modal").find(".module-id").val();

			evaluate(id);
		});
	}
	
	return dialog;
}


/*
 * Deploy the modules
 */

function deployAll(){
	$.post("/dianne/deployer", {"action":"deploy","modules":JSON.stringify(modules)}, 
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
		"module":JSON.stringify(modules[id]),
		"target": target}, 
			function( data ) {
				$.each( data, function(id,target){
					deployment[id] = target;
					// TODO separate color per node?
					$("#"+id).css('background-color', '#FF6CDA');
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

/*
 * Learning functions
 */

function learn(id){
	// first create the chart
	createErrorChart($("#dialog-"+id).find(".error"));

	source = new EventSource("learner");
	source.onmessage = function(event){
		var data = JSON.parse(event.data);
		var index = Number($("#dialog-"+id).find(".error").attr("data-highcharts-chart"));
    	var x = Number(data.sample);
        var y = Number(data.error); 
		Highcharts.charts[index].series[0].addPoint([x, y], true, true, false);
	};
	$.post("/dianne/learner", {"action":"learn",
		"config":JSON.stringify(learning),
		"target": id}, 
			function( data ) {
				$.each(data, function(id, weights){
					modules[id].weights = weights;
				});
				source.close();
			}
			, "json");
}

function evaluate(id){
	// reset chart
	var index = Number($("#dialog-"+id).find(".evaluate").attr("data-highcharts-chart"));
	Highcharts.charts[index].series[0].setData(null, true, true, false);
	$("#dialog-"+id).find(".accuracy").text("");

	source = new EventSource("learner");
	source.onmessage = function(event){
		var data = JSON.parse(event.data);
		var index = Number($("#dialog-"+id).find(".evaluate").attr("data-highcharts-chart"));
		Highcharts.charts[index].series[0].setData(data, true, true, false);
	};
	$.post("/dianne/learner", {"action":"evaluate",
		"config":JSON.stringify(learning),
		"target": id}, 
			function( data ) {
				console.log("DONE!");
				source.close();
				$("#dialog-"+id).find(".accuracy").text("Accuracy: "+data.accuracy+" %");
			}
			, "json");
}

/*
 * SSE for feedback when training/running
 */
var source;

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

/*
 * Helper functions
 */

/**
 * Generates a GUID string.
 * @returns {String} The generated GUID.
 * @example af8a8416-6e18-a307-bd9c-f2c947bbb3aa
 * @author Slavik Meltser (slavik@meltser.info).
 * @link http://slavik.meltser.info/?p=142
 */
function guid() {
    function _p8(s) {
        var p = (Math.random().toString(16)+"000000000").substr(2,8);
        return s ? "-" + p.substr(0,4) + "-" + p.substr(4,4) : p ;
    }
    return _p8() + _p8(true) + _p8(true) + _p8();
}

function renderTemplate(templateId, options){
	var template = $('#'+templateId).html();
	Mustache.parse(template);
	var rendered = Mustache.render(template, options);
	return rendered;
}