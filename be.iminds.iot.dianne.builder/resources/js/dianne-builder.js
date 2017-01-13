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
/**
 * This script allows to create a NN structure by drag-and-drop using jsPlumb
 */

// keep a map of neural network modules, as well as name and id
var nn = {};
nn.name = undefined;
nn.id = undefined;
nn.modules = {};
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
	// remove all dialogs
	$(".modal").remove();
	
	currentMode = m;
	if(currentMode === "build"){
		console.log("switch to build");
		$("#menu-build").addClass("active");
		
		setupBuildToolbox();
	} else if(currentMode === "deploy"){
		console.log("switch to deploy");
		if(nn.name===undefined && Object.keys(nn.modules).length > 0){
			// NN should be saved before deploy
			showSaveDialog();
		} 
			
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
		jsPlumb.repaintEverything();
	} else if(currentMode === "run"){
		console.log("switch to run");
		$("#menu-run").addClass("active");
		
		setupRunToolbox();
		// only show run modules in run mode
		$(".run").each(function( index ) {
			jsPlumb.show($(this).attr('id'),true);
			$(this).show();
		});
		jsPlumb.repaintEverything();
	}
	
	// reset eventsource
	eventsource = undefined;
}

/**
 * Lock UI mode (handy for interactive demos in kiosk mode) on ctrl+k
 */

window.addEventListener("keydown", checkKeyPressed, false);
window.addEventListener("keyup", checkKeyReleased, false);

var ctrl = false;
var locked = false;

function checkKeyPressed(e) {
    if (e.keyCode == "17") {
    	ctrl = true
    } else if(e.keyCode == "75"){
    	if(ctrl){
	    	if(!locked){
	    		$(".controls").hide();
	    		locked = true;
	    	} else {
	    		$(".controls").show();
	    		locked = false;
	    	}
    	}
    }
}

function checkKeyReleased(e) {
    if (e.keyCode == "17") {
    	ctrl = false;
    } 
}

/**
 * On ready, fill the toolbox with available supported modules
 */
$( document ).ready(function() {
	// show correct mode
	setModus(currentMode);
	$("#toolbox").mCustomScrollbar({ theme:"minimal" });
	$("#spinnerwrap").hide();
});


function setupBuildToolbox(){
	$('#toolbox .mCSB_container').empty();
	$.post("/dianne/builder", {action : "available-modules"}, 
			function( data ) {
				$.each(data, function(index, module){
					console.log(module);
					// TODO fetch name/type/category
					if(module.trainable!==undefined){
						trainable[module.type] = "true";
					}
					addToolboxItem(module.type, module.type, module.category, 'build');
				});
			}
			, "json");
}

function setupDeployToolbox(){
	$('#toolbox .mCSB_container').empty();
	$('<p style="margin-left:10px">tags: <input id="tags" type="text" /></p>').appendTo($('#toolbox .mCSB_container'));
	$('<button id="deployAll" class="btn btn-default btn-toolbox controls" onclick="deployAll();return false;">Deploy all</button>').appendTo($('#toolbox .mCSB_container'));
	$('<button id="undeployAll" class="btn btn-default btn-toolbox controls"  onclick="undeployAll();return false;">Undeploy all</button>').appendTo($('#toolbox .mCSB_container'));

	
	selectedTarget = undefined;
	
	$.post("/dianne/deployer", {"action" : "targets"}, 
			function( data ) {
				$.each(data, function(index, target){
					addToolboxItem(target.name, target.id, 'Targets','deploy');
				});
			}
			, "json");
}

function setupLearnToolbox(){
	$('#toolbox .mCSB_container').empty();
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
	addToolboxItem('Evaluator','Evaluator','Evaluator','learn');
}

function setupRunToolbox(){
	$('#toolbox .mCSB_container').empty();
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
					addToolboxItem(output.name, output.type, 'Output', 'run');
				});
			}
			, "json");
	
	$.post("/dianne/input", {action : "available-inputs"}, 
			function( data ) {
				$.each(data, function(index, input){
					addToolboxItem(input.name, input.type, 'Input', 'run');
				});
			}
			, "json");
	
	addToolboxItem('Canvas input','CanvasInput','Input','run');
	addToolboxItem('URL input','URLInput','Input','run');
	addToolboxItem('Raw input','RawInput','Input','run');
	addToolboxItem('Output probabilities','ProbabilityOutput','Visualize','run');
	addToolboxItem('Raw outputs','RawOutput','Visualize','run');
	addToolboxItem('Laserscan output','LaserScan','Visualize','run');
	
	$.post("/dianne/builder", {action : "available-modules"}, 
			function( data ) {
				$.each(data, function(index, module){
					if(module.type==="Memory"){
						addToolboxItem('CharRNN','CharRNN','RNN','run');
					}
				});
			}
			, "json");
}

/**
 * add a toolbox item name to toolbox and add category
 */
function addToolboxItem(name, type, category, mode){
	var panel = $('#toolbox').find('.'+category);
	if(panel.length===0){
		$("<div>"  +
			"<h3 href=\"#\" data-toggle=\"collapse\" data-target=\"."+category+"\">" + category+"</h3>"+	
			"<div class=\""+category+" collapse in row\"></div>" +
		  "</div>").appendTo($('#toolbox .mCSB_container'));
		panel = $('#toolbox').find('.'+category);
	}
	
	var div = $("<div class=\"tool\"></div>").appendTo(panel);
	
	var module = renderTemplate("module",
		{	
			name: name,
			type: type, 
			category: category,
			mode: mode
		}, 
		div);
	

	// make toolbox modules draggable to instantiate using drag-and-drop
	if(mode!=="deploy"){ // not in deploy mode however
		module.draggable({helper: "clone", scroll: false, appendTo: "#builder", containment: '#builder'});
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
	} else {
		var c = deploymentColors[type]; 
		if(c === undefined){
			c = nextColor();
			deploymentColors[type] = c;
		}
		module.css('background-color', c);
		module.css('opacity', 0.5);
		module.click(function() {
			$('#toolbox').find('.module').css('opacity',0.5);
			$(this).css('opacity',0.8);
			selectedTarget = $(this).attr('type');
		});
	}
}


// colors for deployed modules
var selectedTarget;
var deploymentColors = {};
var colors = ['#FF6CDA','#81F781','#AC58FA','#FA5858'];
var colorIndex = 0;

function nextColor(){
	if(colorIndex >= colors.length){
		// generate random color
		return randomColor();
	}
	return colors[colorIndex++];
}

function randomColor() {
    var r, g, b;
    var h = Math.random();
    var i = ~~(h * 6);
    var f = h * 6 - i;
    var q = 1 - f;
    switch(i % 6){
        case 0: r = 1; g = f; b = 0; break;
        case 1: r = q; g = 1; b = 0; break;
        case 2: r = 0; g = 1; b = f; break;
        case 3: r = 0; g = q; b = 1; break;
        case 4: r = f; g = 0; b = 1; break;
        case 5: r = 1; g = 0; b = q; break;
    }
    var c = "#" + ("00" + (~ ~(r * 255)).toString(16)).slice(-2) + ("00" + (~ ~(g * 255)).toString(16)).slice(-2) + ("00" + (~ ~(b * 255)).toString(16)).slice(-2);
    return (c);
}


/*
 * jsPlumb rendering and setup
 */

// definition of source Endpoints
var sourceStyle = {
	isSource:true,
	//anchor : "Right",	
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
	//anchor: "Left",					
	paintStyle:{ 
		fillStyle:"#333"
	},
	hoverPaintStyle:{ 
		fillStyle: "#555"
	}
}

// jsPlumb init code
jsPlumb.ready(function() {       
    jsPlumb.setContainer($("#canvas"));
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

    $('#canvas').on("scroll",function() {
        jsPlumb.repaintEverything();
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
	// should be on the canvas
	if(moduleItem.css('left').startsWith('-')){
		moduleItem.remove();
		jsPlumb.repaintEverything();
		return;
	}
	
	// get type from toolbox item and generate new UUID
	var name = moduleItem.attr("name");
	var type = moduleItem.attr("type");
	var category = moduleItem.attr("category");
	var mode = moduleItem.attr("mode");
	var id = guid();
	moduleItem.attr("id",id);
	
	// setup UI stuff (add to jsPlumb, attach dialog etc)
	setupModule(moduleItem, type, category);

	// create module object
	var module = {};
	module.name = name;
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
		module.test = Math.round(module.total/10);
		module.train = module.total - module.test;
		module.validation = 0;
	} else if(category==="Trainer"){
		module.method = "GD";
		module.batch = 10;
		module.learningRate = 0.01;
		module.momentum = 0.9;
		module.regularization = 0;
		module.loss = "MSE";
		module.clean = false;
	}
	
	// add to one of the module maps
	if(mode==="build"){
		nn.modules[id] = module;
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
		jsPlumb.addEndpoint(moduleItem, sourceStyle, {anchor: "Right"});
		jsPlumb.addEndpoint(moduleItem, targetStyle, {anchor: "Left",endpoint:"Rectangle",filter:":not(.build)",maxConnections:-1});
	} else if(type==="Output"){
		jsPlumb.addEndpoint(moduleItem, sourceStyle, {anchor: "Right",endpoint:"Rectangle", maxConnections:-1});
		jsPlumb.addEndpoint(moduleItem, targetStyle, {anchor: "Left"});
	} else if(category==="Trainer" || category==="Evaluator"){
		jsPlumb.addEndpoint(moduleItem, targetStyle, {anchor: "Left", endpoint:"Rectangle"});
	} else if(category==="Memory"){
		jsPlumb.addEndpoint(moduleItem, sourceStyle, {anchor: "Left"});
		jsPlumb.addEndpoint(moduleItem, targetStyle, {anchor: "Right"});
	} else if(category==="Dataset"){ 
		jsPlumb.addEndpoint(moduleItem, sourceStyle, {anchor: "Right",endpoint:"Rectangle"});
	} else if(category==="Input"){ 
		jsPlumb.addEndpoint(moduleItem, sourceStyle, {anchor: "Right",endpoint:"Rectangle"});
	} else if(category==="Visualize"){ 
		jsPlumb.addEndpoint(moduleItem, targetStyle, {anchor: "Left",endpoint:"Rectangle"});
	} else if(category==="Output"){ 
		jsPlumb.addEndpoint(moduleItem, targetStyle, {anchor: "Left",endpoint:"Rectangle"});
	} else if(category==="RNN"){ 
		jsPlumb.addEndpoint(moduleItem, sourceStyle, {anchor: "Left",endpoint:"Rectangle"});
		jsPlumb.addEndpoint(moduleItem, targetStyle, {anchor: "Right",endpoint:"Rectangle"});
	} else if(category==="Fork" || category==="Variational") {
		jsPlumb.addEndpoint(moduleItem, sourceStyle, {anchor: "Right",maxConnections:-1});
		jsPlumb.addEndpoint(moduleItem, targetStyle, {anchor: "Left"});
	} else if(category==="Join" || category==="Variational") {
		jsPlumb.addEndpoint(moduleItem, sourceStyle, {anchor: "Right"});
		jsPlumb.addEndpoint(moduleItem, targetStyle, {anchor: "Left", maxConnections:-1});
	} else {
		jsPlumb.addEndpoint(moduleItem, sourceStyle, {anchor: "Right"});
		jsPlumb.addEndpoint(moduleItem, targetStyle, {anchor: "Left"});
	}
	
	// show dialog on double click
	moduleItem.dblclick(function() {
		showConfigureModuleDialog($(this));
	});
	
	// add click behavior in deploy mode
	moduleItem.click(function() {
		if(currentMode==='deploy'){
			if(selectedTarget!==undefined){
				var id = $(this).attr('id');
				var target = selectedTarget;
				deploy(id, target);
			}
		}
	});
	
	// make draggable
	moduleItem.draggable(
	{
		drag: function(){
		    jsPlumb.repaintEverything();
		},
		stop: function(ev){
			if($(this).css('left').startsWith('-')){
				removeModule($(this));
			}
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
		if(nn.modules[id].next!==undefined){
			$.each(nn.modules[id].next, function( index, next ) {
				removePrevious(next, id);
			});
		}
		if(nn.modules[id].prev!==undefined){
			$.each(nn.modules[id].prev, function(index, prev){
				removeNext(prev, id);
			});
		}
		delete nn.modules[id];
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
	if(nn.modules[connection.sourceId]===undefined){
		if(learning[connection.sourceId]!==undefined){
			learning[connection.sourceId].input = connection.targetId; 
		} else {
			running[connection.sourceId].input = connection.targetId;
			$.post("/dianne/input", {"action" : "setinput",
				"nnId" : nn.id,
				"inputId" : connection.targetId,
				"input" : running[connection.sourceId].name});
		}
	} else if(nn.modules[connection.targetId]===undefined){
		if(learning[connection.targetId]!==undefined){
			learning[connection.targetId].output = connection.sourceId; 
		} else {
			running[connection.targetId].output = connection.sourceId;
			$.post("/dianne/output", {"action" : "setoutput",
				"nnId" : nn.id,
				"outputId" : connection.sourceId,
				"output" : running[connection.targetId].name});
		}
	} else {
		addNext(connection.sourceId, connection.targetId);
		addPrevious(connection.targetId, connection.sourceId);
		
		connection.connection.bind("dblclick", connectionClicked);
	}
}

/**
 * Remove a connection between two modules
 * @param connection to remove
 */
function removeConnection(connection){
	console.log("Remove connection " + connection.sourceId + " -> " + connection.suspendedElementId);
	// TODO support multiple next/prev
	if(nn.modules[connection.sourceId]===undefined){
		if(learning[connection.sourceId]!==undefined){
			delete learning[connection.sourceId].input; 
		} else {
			delete running[connection.sourceId].input; 
			$.post("/dianne/input", {"action" : "unsetinput",
				"nnId" : nn.id,
				"inputId" : connection.suspendedElementId,
				"input" : running[connection.sourceId].name});
		}
	} else if(nn.modules[connection.suspendedElementId]===undefined){
		if(learning[connection.suspendedElementId]!==undefined){
			delete learning[connection.suspendedElementId].output; 
		} else {
			delete running[connection.suspendedElementId].output;
			$.post("/dianne/output", {"action" : "unsetoutput",
				"nnId" : nn.id,
				"outputId" : connection.sourceId,
				"output" : running[connection.suspendedElementId].name});
		}
	} else {
		removeNext(connection.sourceId, connection.suspendedElementId);	
		removePrevious(connection.suspendedElementId, connection.sourceId);
	}
}

function addNext(id, next){
	if(nn.modules[id].next === undefined){
		nn.modules[id].next = [next];
	} else { 
		nn.modules[id].next.push(next);
	}
}

function removeNext(id, next){
	if(nn.modules[id].next.length==1){
		delete nn.modules[id].next;
	} else {
		var i = nn.modules[id].next.indexOf(next);
		nn.modules[id].next.splice(i, 1);
	} 
}

function addPrevious(id, prev){
	if(nn.modules[id].prev === undefined){
		nn.modules[id].prev = [prev];
	} else { 
		nn.modules[id].prev.push(prev);
	}
}

function removePrevious(id, prev){
	if(nn.modules[id].prev.length==1){	
		delete nn.modules[id].prev;
	} else {
		var i = nn.modules[id].prev.indexOf(prev);
		nn.modules[id].prev.splice(i, 1);
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
 * Save, load and recover 
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
			value: nn.name
		}, dialog.find('.form-items'));

	renderTemplate('form-checkbox',
			{
				name: "New uuids",
				id: "copy",
				checked: ""
			}, dialog.find('.form-items'));
	
	// submit button callback
	dialog.find(".submit").click(function(e){
		var name = $(this).closest('.modal').find('.name').val();
		var copy = $(this).closest('.modal').find('.copy').is(':checked');

		save(name, copy);
	});
	
	// remove cancel button
	dialog.find('.cancel').remove();
	// remove module-modal specific stuff
	dialog.removeClass("module-modal");
	dialog.find('.module-dialog').removeClass("module-dialog");
	// show dialog
	dialog.modal('show');
	
}

function save(name, copy){
	nn.name = name;
	
	// save modules
	var s = {};
	s.modules = nn.modules;
	s.name = name;
	
	var modulesJson = JSON.stringify(s);
	
	// save layout
	var layout = saveLayout();
    var layoutJson = JSON.stringify(layout);
    
    if(copy){
    	$.each(nn.modules, function( k, v ) {
    		var newId = guid();
    		modulesJson = modulesJson.replace(k, newId, "g");
    		layoutJson = layoutJson.replace(k, newId, "g");
    	});
    	
    }
    
	$.post("/dianne/save", {"nn":modulesJson, "layout":layoutJson}, 
		function( data ) {
			$('#dialog-save').remove();
			console.log("saved "+s.name);
			load(s.name);
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
				data.sort();
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
	console.log("load "+name);
	$.post("/dianne/load", {"action":"load", "name":name}, 
			function( data ) {
				resetCanvas();
		
				nn.name = name;
				$('#name').text(nn.name);
				
				nn.modules = data.nn.modules;
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
        var connection = jsPlumb.connect({
        	source: elem.sourceId,
        	target: elem.targetId,
        	anchors: elem.anchors
        });
        
        if(connection!==undefined)
        	connection.bind("dblclick", connectionClicked);
    });
}

function redrawElement(id, posX, posY){
	var module = nn.modules[id];

	var moduleItem = renderTemplate("module",
			{	
				name: module.name,
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


function connectionClicked(connection){
	console.log("Connection clicked "+connection.sourceId+"->"+connection.targetId);
	showConnectionDialog(connection);
}


function showRecoverDialog(){
	var dialog = renderTemplate("dialog", {
		id : "recover",
		title : "Recover a neural network ",
		submit: "Recover",
		cancel: "Cancel"
	}, $(document.body));
	
	dialog.find('.content').append("<p>Select a neural network to recover.</p>")
	
	renderTemplate("form-dropdown", 
			{	
				name: "Neural network: "
			},
			dialog.find('.form-items'));
	$.post("/dianne/deployer", {"action" : "recover"}, 
			function( data ) {
				$.each(data, function(index, nn){
					dialog.find('.options').append("<option value="+nn.id+">"+nn.name+" ("+nn.description+")</option>")
				});
			}
			, "json");
	
	// submit button callback
	dialog.find(".submit").click(function(e){
		var id = $(this).closest('.modal').find('.options').val();
		recover(id);
	});
	
	// remove cancel button
	dialog.find('.cancel').remove();
	// remove module-modal specific stuff
	dialog.removeClass("module-modal");
	dialog.find('.module-dialog').removeClass("module-dialog");
	// show dialog
	dialog.modal('show');
	
}

function recover(id){
	console.log("recover "+id);
	
	$.post("/dianne/deployer", {"action":"recover", "id":id}, 
			function( data ) {
				// empty canvas?
				resetCanvas();
				
				nn.name = data.nn.name;
				$('#name').text(nn.name);
				nn.id = data.id;
				nn.modules = data.nn.modules;
				loadLayout(data.layout);
				deployment = data.deployment;
				$.each( data.deployment, color );
				$('#dialog-recover').remove();
				console.log("Succesfully recovered");
			}
			, "json");
}

function resetCanvas(){
	// empty canvas - keep alerts and name div
	var name = $('#canvas #name').detach();
	var alerts = $('#canvas #alerts').detach();
	$('#canvas').empty().append(name).append(alerts);
	learning = {};
	running = {};
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


/**
 * Redirect to dashboard if exists
 */
function redirect(){
	$.ajax({
	    type: 'HEAD',
	    url: '../dashboard/dashboard.html',
	    	success: function() {
	    		window.location = '../dashboard/dashboard.html'
	    	},
	    	error: function() {
	    	}
	});
}