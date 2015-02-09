/**
 * This script allows to create a NN structure by drag-and-drop using jsPlumb
 */

// keep a map of neural network modules
var nn = {};
// keep a map of all learn blocks
var learning = {};
// keep a map of all run blocks
var running = {};

// keep map module id -> deployment node
var deployment = {};

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
		$(".toolbox").hide();
		$("#menu-build").addClass("active");
		$("#toolbox-build").show();
		$("#toolbox-build").addClass("active");
	} else if(currentMode === "deploy"){
		console.log("switch to deploy");
		$(".toolbox").hide();
		$("#menu-deploy").addClass("active");
		$("#toolbox-deploy").show();
		$("#toolbox-deploy").addClass("active");
	} else if(currentMode === "learn"){
		console.log("switch to learn");
		$(".toolbox").hide();
		$("#menu-learn").addClass("active");
		$("#toolbox-learn").show();
		$("#toolbox-learn").addClass("active");
		// only show learn modules in learn mode
		$(".learn").each(function( index ) {
			jsPlumb.show($(this).attr('id'),true);
			$(this).show();
		});
	} else if(currentMode === "run"){
		console.log("switch to run");
		$(".toolbox").hide();
		$("#menu-run").addClass("active");
		$("#toolbox-run").show();
		$("#toolbox-run").addClass("active");
		// only show run modules in learn mode
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
	// initialize toolboxes
	// build toolbox
	$.post("/dianne/builder", {action : "available-modules"}, 
		function( data ) {
			$.each(data, function(index, type){
				// TODO fetch name/type/category
				addToolboxItem('toolbox-build', type, type, type, 'build');
			});
		}
		, "json");
	
	// learn toolbox
	// TODO this is hard coded for now, as this does not map to factories/module impls
	addToolboxItem('toolbox-learn','MNIST Dataset','MNIST','Dataset','learn');
	addToolboxItem('toolbox-learn','SGD Trainer','StochasticGradientDescent','Trainer','learn');
	addToolboxItem('toolbox-learn','Arg Max Evaluator','ArgMax','Evaluator','learn');
	
	addToolboxItem('toolbox-run','MNIST input','MNIST','Dataset','run');
	addToolboxItem('toolbox-run','Canvas input','CanvasInput','RunInput','run');
	addToolboxItem('toolbox-run','Output probabilities','ProbabilityOutput','RunOutput','run');
	
	// show correct mode
	setModus(currentMode);
});


/**
 * add a toolbox item name to toolbox with id toolboxId and add category
 */
function addToolboxItem(toolboxId, name, type, category, mode){

	var module = renderTemplate("module",
		{	
			name: name,
			type: type, 
			category: category,
			mode: mode
		}, 
		$('#'+toolboxId));
	

			// make toolbox modules draggable to instantiate using drag-and-drop
	module.draggable({helper: "clone"});
	module.bind('dragstop', function(event, ui) {
		if(checkAddModule($(this))){
			// clone the toolbox item
		    var moduleItem = $(ui.helper).clone().addClass(mode);
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
 * Module/Connection add/remove methods
 */

/**
 * Add a module to the canvas and to modules datastructure
 * 
 * @param moduleItem a freshly cloned DOM element from toolbox item 
 * @param toolboxItem the toolbox DOM element the moduleItem was cloned from
 */
function addModule(moduleItem){
	// only if comes from toolbox, could also be loaded from file
	moduleItem.appendTo("#canvas");
		
	// fix offset of toolbox 
	var offset = {};
	offset.left = moduleItem.offset().left - ($("#canvas").offset().left - $(".toolbox.active").offset().left);
	offset.top = moduleItem.offset().top - ($("#canvas").offset().top - $(".toolbox.active").offset().top);
	moduleItem.offset(offset);
	  
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
	
	// some hard coded shit here... should be changed
	if(category==="Dataset"){
		// TODO this is hard coded for MNIST
		module.dataset = "MNIST";
		module.total = 70000;
		module.train = 60000;
		module.test = 10000;
		module.validation = 0;
		
	} else if(category==="Trainer"){
		// TODO this is hard coded
		//module.strategy = "Stochastic Gradient Descent";
		module.batch = 10;
		module.epochs = 1;
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
	} else if(category==="RunInput"){ 
		jsPlumb.addEndpoint(moduleItem, sourceStyle, {endpoint:"Rectangle"});
	} else if(category==="RunOutput"){ 
		jsPlumb.addEndpoint(moduleItem, targetStyle, {endpoint:"Rectangle"});
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
	if(mode==="nn"){
		if(nn[nn[id].next]!==undefined){
			delete nn[nn[id].next].prev;
		}
		if(nn[nn[id].prev]!==undefined){
			delete nn[nn[id].prev].next;
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
		}
	} else {
		nn[connection.sourceId].next = connection.targetId;
		nn[connection.targetId].prev = connection.sourceId;
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
		}
	} else {
		delete nn[connection.sourceId].next;
		delete nn[connection.targetId].prev;
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
		if(connection.connection.endpoints[0].type!=="Dot" 
			|| connection.connection.endpoints[1].type!=="Dot"){
				return false;
		}
	}
	if(currentMode==="learn"){
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
	if(currentMode==="build"){
		if(connection.connection.endpoints[0].type!=="Dot" 
			|| connection.connection.endpoints[1].type!=="Dot"){
				return false;
		}
		if(deployment[connection.sourceId]!==undefined
				|| deployment[connection.targetId]!==undefined){
				return false;
		}
	}
	if(currentMode==="learn" || currentMode==="run"){
		if(connection.connection.endpoints[0].type!=="Rectangle" 
			|| connection.connection.endpoints[1].type!=="Rectangle"){
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
	console.log("save");
	// save modules
	var modulesJson = JSON.stringify(nn);
	
	// save layout
	var layout = saveLayout();
    var layoutJson = JSON.stringify(layout);
    console.log(layoutJson);
    
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
