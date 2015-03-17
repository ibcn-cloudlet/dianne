/**
 * This script allows to create a NN structure by drag-and-drop using jsPlumb
 */

// keep a map of neural network modules
var nn = {};
// keep a map of all learn blocks
var learning = {};
// keep a map of all run blocks
var running = {};

// keep a map of known datasets
var datasets = {};

// keep map module id -> deployment node
var deployment = {};

// keep a map of all trainable module types
var trainable = {};

/*
 * UI Mode
 */
var currentMode = "build";

/**
 * Set UI modus (build / deploy / learn / run )
 */
function setModus(m){
	$(".active").removeClass("active");
	// only show learn/run modules in learn/run mode
	$(".learn").each(function( index ) {
		jsPlumb.hide($(this).attr('id'),true);
		$(this).hide();
	});
	$(".run").each(function( index ) {
		jsPlumb.hide($(this).attr('id'),true);
		$(this).hide();
	});
	currentMode = m;
	if(currentMode === "build"){
		console.log("switch to build");
		$("#menu-build").addClass("active");
		
		setupBuildToolbox();
	} else if(currentMode === "deploy"){
		console.log("switch to deploy");
		$("#menu-deploy").addClass("active");

		setupDeployToolbox();
	} else if(currentMode === "learn"){
		console.log("switch to learn");
		$("#menu-learn").addClass("active");

		setupLearnToolbox();
		// only show learn modules in learn mode
		$(".learn").each(function( index ) {
			jsPlumb.show($(this).attr('id'),true);
			$(this).show();
		});
	} else if(currentMode === "run"){
		console.log("switch to run");
		$("#menu-run").addClass("active");
		
		setupRunToolbox();
		// only show run modules in run mode
		$(".run").each(function( index ) {
			jsPlumb.show($(this).attr('id'),true);
			$(this).show();
		});
	}
	// remove all dialogs
	$(".modal").remove();
}

/**
 * On ready, fill the toolbox with available supported modules
 */
$( document ).ready(function() {
	// show correct mode
	setModus(currentMode);
});


function setupBuildToolbox(){
	$('#toolbox').empty();
	$.post("/dianne/builder", {action : "available-modules"}, 
			function( data ) {
				$.each(data, function(index, module){
					console.log(module);
					// TODO fetch name/type/category
					if(module.trainable!==undefined){
						trainable[module.type] = true;
					}
					addToolboxItem(module.type, module.type, module.category, 'build');
				});
			}
			, "json");
}

function setupDeployToolbox(){
	$('#toolbox').empty();
	$('<button id="deployAll" class="btn btn-default" onclick="deployAll();return false;">Deploy all</button>').appendTo($('#toolbox'));
	$('<button id="undeployAll" class="btn btn-default"  onclick="undeployAll();return false;">Undeploy all</button>').appendTo($('#toolbox'));
}

function setupLearnToolbox(){
	$('#toolbox').empty();
	$.post("/dianne/datasets", {action : "available-datasets"}, 
			function( data ) {
				$.each(data, function(index, dataset){
					// add datasets to learn/run toolboxes
					datasets[dataset.dataset] = dataset;
					addToolboxItem(dataset.dataset, dataset.dataset, 'Dataset', 'learn');
				});
			}
			, "json");
	
	addToolboxItem('SGD Trainer','StochasticGradientDescent','Trainer','learn');
	addToolboxItem('Arg Max Evaluator','ArgMax','Evaluator','learn');
}

function setupRunToolbox(){
	$('#toolbox').empty();
	$.post("/dianne/datasets", {action : "available-datasets"}, 
			function( data ) {
				$.each(data, function(index, dataset){
					// add datasets to learn/run toolboxes
					datasets[dataset.dataset] = dataset;
					addToolboxItem(dataset.dataset, dataset.dataset, 'Dataset', 'run');
				});
			}
			, "json");
	
	$.post("/dianne/output", {action : "available-outputs"}, 
			function( data ) {
				$.each(data, function(index, output){
					addToolboxItem(output, output, 'Output', 'run');
				});
			}
			, "json");
	
	addToolboxItem('Canvas input','CanvasInput','Source','run');
	addToolboxItem('Output probabilities','ProbabilityOutput','Visualize','run');
}

/**
 * add a toolbox item name to toolbox and add category
 */
function addToolboxItem(name, type, category, mode){
	var panel = $('#toolbox').find('.'+category);
	if(panel.length===0){
		$("<div>"  +
			"<h4 data-toggle=\"collapse\" data-target=\"."+category+"\">"+category+"</h4>"+	
			"<div class=\""+category+" collapse in row\"></div>" +
		  "</div>").appendTo($('#toolbox'));
		panel = $('#toolbox').find('.'+category);
	}
	
	var div = $("<div class=\"tool col-xs-6\"></div>").appendTo(panel);
	
	var module = renderTemplate("module",
		{	
			name: name,
			type: type, 
			category: category,
			mode: mode
		}, 
		div);
	

	// make toolbox modules draggable to instantiate using drag-and-drop
	module.draggable({helper: "clone"});
	module.bind('dragstop', function(event, ui) {
		if(checkAddModule($(this))){
			// clone the toolbox item
		    var moduleItem = $(ui.helper).clone().addClass(mode);
		    
			// append to canvas
			moduleItem.appendTo("#canvas");
		    
		    // fix offset after drag
			moduleItem.offset(ui.offset);
		    
			// add module
			addModule(moduleItem);
		}
	});

	
}

/*
 * jsPlumb rendering and setup
 */

// definition of source Endpoints
var sourceStyle = {
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
var targetStyle = {
	isTarget:true,
	anchor: "Left",					
	paintStyle:{ 
		fillStyle:"#333"
	},
	hoverPaintStyle:{ 
		fillStyle: "#555"
	}
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
			if(!checkAddConnection(connection.connection)){
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
 * Module/Connection add/remove methods
 */

/**
 * Add a module to the canvas and to modules datastructure
 * 
 * @param moduleItem a freshly cloned DOM element from toolbox item 
 * @param toolboxItem the toolbox DOM element the moduleItem was cloned from
 */
function addModule(moduleItem){

	// get type from toolbox item and generate new UUID
	var type = moduleItem.attr("type");
	var category = moduleItem.attr("category");
	var mode = moduleItem.attr("mode");
	var id = guid();
	moduleItem.attr("id",id);
	
	// setup UI stuff (add to jsPlumb, attach dialog etc)
	setupModule(moduleItem, type, category);

	// create module object
	var module = {};
	module.type = type;
	module.category = category;
	module.id = id;
	if(trainable[type]!==undefined){
		module.trainable = trainable[module.type];
	}
	
	// some hard coded shit here... should be changed
	if(category==="Dataset"){
		module.dataset = module.type;
		module.total = datasets[module.type].size;
		module.labels = datasets[module.type].labels;
		module.test = 10000;
		module.train = module.total - 10000;
		module.validation = 0;
	} else if(category==="Trainer"){
		// TODO this is hard coded
		//module.strategy = "Stochastic Gradient Descent";
		module.batch = 10;
		module.epochs = 1;
		module.learningRate = 0.5;
		module.learningRateDecay = 0.0;
		module.loss = "MSE";
	}
	
	// add to one of the module maps
	if(mode==="build"){
		nn[id] = module;
	} else if(mode==="learn"){
		learning[id] = module;
	} else if(mode==="run"){
		running[id] = module;
	}
	
	console.log("Add module "+id);
}

/**
 * setup module jsPlumb endpoints and drag/click behavior
 */
function setupModule(moduleItem, type, category){
	// TODO this should not be hard coded?
	if(type==="Input"){
		jsPlumb.addEndpoint(moduleItem, sourceStyle);
		jsPlumb.addEndpoint(moduleItem, targetStyle, {endpoint:"Rectangle",filter:":not(.build)",maxConnections:-1});
	} else if(type==="Output"){
		jsPlumb.addEndpoint(moduleItem, sourceStyle, {endpoint:"Rectangle", maxConnections:-1});
		jsPlumb.addEndpoint(moduleItem, targetStyle);
	} else if(category==="Trainer" || category==="Evaluator"){
		jsPlumb.addEndpoint(moduleItem, targetStyle, {endpoint:"Rectangle"});
	} else if(category==="Dataset"){ 
		jsPlumb.addEndpoint(moduleItem, sourceStyle, {endpoint:"Rectangle"});
	} else if(category==="Source"){ 
		jsPlumb.addEndpoint(moduleItem, sourceStyle, {endpoint:"Rectangle"});
	} else if(category==="Visualize"){ 
		jsPlumb.addEndpoint(moduleItem, targetStyle, {endpoint:"Rectangle"});
	} else if(category==="Output"){ 
		jsPlumb.addEndpoint(moduleItem, targetStyle, {endpoint:"Rectangle"});
	} else if(category==="Fork") {
		jsPlumb.addEndpoint(moduleItem, sourceStyle, {maxConnections:-1});
		jsPlumb.addEndpoint(moduleItem, targetStyle);
	} else if(category==="Join") {
		jsPlumb.addEndpoint(moduleItem, sourceStyle);
		jsPlumb.addEndpoint(moduleItem, targetStyle, {maxConnections:-1});
	} else {
		jsPlumb.addEndpoint(moduleItem, sourceStyle);
		jsPlumb.addEndpoint(moduleItem, targetStyle);
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
}

/**
 * Remove a module from the canvas and the modules datastructure
 * 
 * @param moduleItem the DOM element on the canvas representing the module
 */
function removeModule(moduleItem){
	var id = moduleItem.attr("id");
	var mode = moduleItem.attr("mode");

	// delete this moduleItem
	$.each(jsPlumb.getEndpoints(moduleItem), function(index, endpoint){
		jsPlumb.deleteEndpoint(endpoint)}
	);
	
	jsPlumb.detachAllConnections(moduleItem);
	moduleItem.remove();

	// remove from modules
	if(mode==="build"){
		if(nn[id].next!==undefined){
			$.each(nn[id].next, function( index, next ) {
				removePrevious(next, id);
			});
		}
		if(nn[id].prev!==undefined){
			$.each(nn[id].prev, function(index, prev){
				removeNext(prev, id);
			});
		}
		delete nn[id];
	} else if(mode==="learn"){
		delete learning[id];
	} else if(mode==="run"){
		delete running[id];
	}
	console.log("Remove module "+id);
	
}

/**
 * Add a connection between two modules
 * @param connection to add
 */
function addConnection(connection){
	console.log("Add connection " + connection.sourceId + " -> " + connection.targetId);
	// TODO support multiple next/prev
	if(nn[connection.sourceId]===undefined){
		if(learning[connection.sourceId]!==undefined){
			learning[connection.sourceId].input = connection.targetId; 
		} else {
			running[connection.sourceId].input = connection.targetId; 
		}
	} else if(nn[connection.targetId]===undefined){
		if(learning[connection.targetId]!==undefined){
			learning[connection.targetId].output = connection.sourceId; 
		} else {
			running[connection.targetId].output = connection.sourceId;
			$.post("/dianne/output", {action : "setoutput",
				outputId : connection.sourceId,
				output : running[connection.targetId].type});
		}
	} else {
		addNext(connection.sourceId, connection.targetId);
		addPrevious(connection.targetId, connection.sourceId);
	}
}

/**
 * Remove a connection between two modules
 * @param connection to remove
 */
function removeConnection(connection){
	console.log("Remove connection " + connection.sourceId + " -> " + connection.targetId);
	// TODO support multiple next/prev
	if(nn[connection.sourceId]===undefined){
		if(learning[connection.sourceId]!==undefined){
			delete learning[connection.sourceId].input; 
		} else {
			delete running[connection.sourceId].input; 
		}
	} else if(nn[connection.targetId]===undefined){
		if(learning[connection.targetId]!==undefined){
			delete learning[connection.targetId].output; 
		} else {
			delete running[connection.targetId].output;
			$.post("/dianne/output", {action : "unsetoutput",
				outputId : connection.sourceId,
				output : running[connection.targetId].type});
		}
	} else {
		removeNext(connection.sourceId, connection.targetId);	
		removePrevious(connection.targetId, connection.sourceId);
	}
}

function addNext(id, next){
	if(nn[id].next === undefined){
		nn[id].next = [next];
	} else { 
		nn[id].next.push(next);
	}
}

function removeNext(id, next){
	if(nn[id].next.length==1){
		delete nn[id].next;
	} else {
		var i = nn[id].next.indexOf(next);
		nn[id].next.splice(i, 1);
	} 
}

function addPrevious(id, prev){
	if(nn[id].prev === undefined){
		nn[id].prev = [prev];
	} else { 
		nn[id].prev.push(prev);
	}
}

function removePrevious(id, prev){
	if(nn[id].prev.length==1){	
		delete nn[id].prev;
	} else {
		var i = nn[id].prev.indexOf(prev);
		nn[id].prev.splice(i, 1);
	}
}


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
	if(currentMode==="build"){
		if(deployment[connection.sourceId]!==undefined
				|| deployment[connection.targetId]!==undefined){
				return false;
		}
		if(connection.endpoints[0].type!=="Dot" 
			|| connection.endpoints[1].type!=="Dot"){
				return false;
		}
	}
	if(currentMode==="learn"){
		if(connection.endpoints[0].type!=="Rectangle" 
			|| connection.endpoints[1].type!=="Rectangle"){
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
	if(currentMode==="build"){
		if(connection.endpoints[0].type!=="Dot" 
			|| connection.endpoints[1].type!=="Dot"){
				return false;
		}
		if(deployment[connection.sourceId]!==undefined
				|| deployment[connection.targetId]!==undefined){
				return false;
		}
	}
	if(currentMode==="learn" || currentMode==="run"){
		if(connection.endpoints[0].type!=="Rectangle" 
			|| connection.endpoints[1].type!=="Rectangle"){
				return false;
		}
	}
	return true;
}



/*
 * Save and load 
 */

function showSaveDialog(){
	var dialog = renderTemplate("dialog", {
		id : "save",
		title : "Save your neural network ",
		submit: "Save",
		cancel: "Cancel"
	}, $(document.body));
	
	dialog.find('.content').append("<p>Provide a name for your neural network.</p>")
	
	renderTemplate('form-item',
		{
			name: "Name",
			id: "name",
			value: ""
		}, dialog.find('.form-items'));
	
	// submit button callback
	dialog.find(".submit").click(function(e){
		var name = $(this).closest('.modal').find('.form-control').val();
		save(name);
	});
	
	// remove cancel button
	dialog.find('.cancel').remove();
	// remove module-modal specific stuff
	dialog.removeClass("module-modal");
	dialog.find('.module-dialog').removeClass("module-dialog");
	// show dialog
	dialog.modal('show');
	
}

function save(name){
	// save modules
	var modulesJson = JSON.stringify(nn);
	
	// save layout
	var layout = saveLayout();
    var layoutJson = JSON.stringify(layout);
    
	$.post("/dianne/save", {"name":name, "modules":modulesJson, "layout":layoutJson}, 
		function( data ) {
			$('#dialog-save').remove();
			console.log("Succesfully saved");
		}
		, "json");
    
}

function saveLayout(){
	// nodes
    var nodes = []
    $(".build").each(function (idx, elem) {
        var $elem = $(elem);
        var endpoints = jsPlumb.getEndpoints($elem.attr('id'));
        nodes.push({
        	id: $elem.attr('id'),
            positionX: parseInt($elem.css("left"), 10),
            positionY: parseInt($elem.css("top"), 10)
        });
    });
    // connections
    var connections = [];
    $.each(jsPlumb.getConnections(), function (idx, connection) {
        connections.push({
        connectionId: connection.id,
        sourceId: connection.sourceId,
        targetId: connection.targetId,
        // anchors
        anchors: $.map(connection.endpoints, function(endpoint) {

          return [[endpoint.anchor.x, 
                   endpoint.anchor.y, 
                   endpoint.anchor.orientation[0], 
                   endpoint.anchor.orientation[1],
                   endpoint.anchor.offsets[0],
                   endpoint.anchor.offsets[1]]];

	        })
	    });
    });
    
    var layout = {};
    layout.nodes = nodes;
    layout.connections = connections;
    
    return layout;
}

function showLoadDialog(){
	var dialog = renderTemplate("dialog", {
		id : "load",
		title : "Load a neural network ",
		submit: "Load",
		cancel: "Cancel"
	}, $(document.body));
	
	dialog.find('.content').append("<p>Select a neural network to load.</p>")
	
	renderTemplate("form-dropdown", 
			{	
				name: "Neural network: "
			},
			dialog.find('.form-items'));
	$.post("/dianne/load", {"action" : "list"}, 
			function( data ) {
				$.each(data, function(index, name){
					dialog.find('.options').append("<option value="+name+">"+name+"</option>")
				});
			}
			, "json");
	
	// submit button callback
	dialog.find(".submit").click(function(e){
		var name = $(this).closest('.modal').find('.options').val();
		load(name);
	});
	
	// remove cancel button
	dialog.find('.cancel').remove();
	// remove module-modal specific stuff
	dialog.removeClass("module-modal");
	dialog.find('.module-dialog').removeClass("module-dialog");
	// show dialog
	dialog.modal('show');
	
}

function load(name){
	console.log("load");
	
	$.post("/dianne/load", {"action":"load", "name":name}, 
			function( data ) {
				// empty canvas?
				$('#canvas').empty();
		
				nn = data.modules;
				loadLayout(data.layout);
		
				$('#dialog-load').remove();
				console.log("Succesfully loaded");
			}
			, "json");
}

function loadLayout(layout){
    var nodes = layout.nodes;
    $.each(nodes, function( index, elem ) {
    	console.log(elem.id+", "+elem.positionX+", "+elem.positionY);
    	redrawElement(elem.id, elem.positionX, elem.positionY);
    });
    
    var connections = layout.connections;
    $.each(connections, function( index, elem ) {
        var connection1 = jsPlumb.connect({
        	source: elem.sourceId,
        	target: elem.targetId,
        	anchors: elem.anchors
        });
    });
}

function redrawElement(id, posX, posY){
	var module = nn[id];

	var moduleItem = renderTemplate("module",
			{	
				name: module.type,
				type: module.type, 
				category: module.category,
				mode: "build"
			}, 
			$('#canvas'));
	
	moduleItem.addClass("build");
	moduleItem.attr("id", id);
	moduleItem.draggable();
	moduleItem.css('position','absolute');
	moduleItem.css('left', posX);
	moduleItem.css('top', posY);
	
	setupModule(moduleItem, module.type, module.category);
	jsPlumb.repaint(id);
}

/*
 * Helper functions
 */

/**
 * render a template and append to the target
 */
function renderTemplate(template, options, target){
	var template = $('#'+template).html();
	Mustache.parse(template);
	var rendered = Mustache.render(template, options);
	return $(rendered).appendTo(target);
}

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
