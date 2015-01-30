/**
 * This script allows to create a NN structure by drag-and-drop using jsPlumb
 */

// keep a model of constructed modules
var modules = {};


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
	},
//		maxConnections:-1,
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
	},
//		maxConnections:-1,
}

/**
 * On ready, fill the toolbox with available supported modules
 */
$( document ).ready(function() {
	$.post("/dianne/builder", {action : "available-modules"}, 
		function( data ) {
			$.each(data, function(index, name){
				console.log(name);	
				// Render toolbox item
				$('#toolbox').append(renderTemplate("toolbox-module",
						{name: name }));
				
				// make draggable and add code to create new modules drag-and-drop style
				$('#'+name).draggable({helper: "clone"});
				$('#'+name).bind('dragstop', function(event, ui) {
					if(checkAddModule($(this))){
						// clone the toolbox item
					    var moduleItem = $(ui.helper).clone().removeClass("toolbox");
					    
						addModule(moduleItem, $(this));
					}
				});
			});
		}
		, "json");
});

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
 * Module configuration/deletion dialog stuff 
 */

function showConfigureModuleDialog(moduleItem){
	var id = moduleItem.attr("id");
	
	// there can be only one dialog at a time for one module
	// try to reuse dialog
	var dialogId = "dialog-"+id;
	var d;
	d = $("#"+dialogId);
	if(d.length==0){
		// create new dialog
		var dialog = renderTemplate("dialog", {
			id : id,
			title : "Configure module"
		});
		
		// TODO check which "mode" you are in, for now only "build" mode
		
		d = createBuildModuleDialog(id, $(dialog));
	}
	
	var offset = moduleItem.offset();
	offset.top = offset.top - 100;
	offset.left = offset.left - 200;
	
	// show the modal (disable backdrop)
	d.modal({'show':true, 'backdrop':false}).draggable({handle: ".modal-header"}).offset(offset);
	
}

function createBuildModuleDialog(id, dialog){
	var module = modules[id];
	
	// create build body form
	var body = renderTemplate("build-dialog-body", {
		id : module.id,
		type : module.type
	});
	dialog.find(".modal-body").append(body);
	
	// then fill in properties
	$.post("/dianne/builder", {"action" : "module-properties","type" : module.type}, 
		function( data ) {
			$.each(data, function(index, property){
				console.log(property);	
				// Render toolbox item
				dialog.find('.form-properties').append(
						renderTemplate("property-form", 
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
	var buttons = renderTemplate("build-dialog-buttons", {});
	dialog.find(".modal-footer").append(buttons);
	
	// add button callbacks
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
		var test = $(this).closest(".modal");
		var test2 = test.find(".module-id");
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
	return true;
}

/**
 * Check whether one is allowed to remove this connection
 */
function checkRemoveConnection(connection){
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
    offset.left = moduleItem.offset().left - ($("#canvas").offset().left - $("#toolbox").offset().left);
    offset.top = moduleItem.offset().top - ($("#canvas").offset().top - $("#toolbox").offset().top);
    moduleItem.offset(offset);
  
    // get type from toolbox item and generate new UUID
	var type = toolboxItem.attr("id");
	var id = guid();
	moduleItem.attr("id",id);
	
	// TODO this should not be hard coded?
	if(type==="Input"){
		jsPlumb.addEndpoint(moduleItem, source);
	} else if(type==="Output"){
		jsPlumb.addEndpoint(moduleItem, target);
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
	modules[id] = module;
	
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
	if(modules[modules[id].next]!==undefined){
		delete modules[modules[id].next].prev;
	}
	if(modules[modules[id].prev]!==undefined){
		delete modules[modules[id].prev].next;
	}
	delete modules[id];
	
	console.log("Remove module "+id);
	
}

/**
 * Add a connection between two modules
 * @param connection to add
 */
function addConnection(connection){
	console.log("Add connection " + connection.sourceId + " -> " + connection.targetId);

	modules[connection.sourceId].next = connection.targetId;
	modules[connection.targetId].prev = connection.sourceId;
}

/**
 * Remove a connection between two modules
 * @param connection to remove
 */
function removeConnection(connection){
	console.log("Remove connection " + connection.sourceId + " -> " + connection.targetId);

	delete modules[connection.sourceId].next;
	delete modules[connection.targetId].prev;
}




/*
 * Deploy the modules
 */

function deploy(){
	$.post("/dianne/deployer", {"modules":JSON.stringify(modules)}, 
			function( data ) {
				// Do something on return?
			}
			, "json");
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