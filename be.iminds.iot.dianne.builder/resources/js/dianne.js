/**
 * This script allows to create a NN structure by drag-and-drop using jsPlumb
 */

// keep a model of constructed modules
var modules = {};

// keep map module id -> deployment node
var deployment = {};

/*
 * UI Mode
 */
var modus = "build";

function setModus(m){
	$(".active").removeClass("active");
	modus = m;
	if(modus === "build"){
		console.log("switch to build");
		$(".toolbox").hide();
		$("#menu-build").addClass("active");
		$("#toolbox-build").show();
		
	} else if(modus === "learn"){
		console.log("switch to learn");
		$(".toolbox").hide();
		$("#menu-learn").addClass("active");
		$("#toolbox-learn").show();
	} else if(modus === "deploy"){
		console.log("switch to deploy");
		$(".toolbox").hide();
		$("#menu-deploy").addClass("active");
		$("#toolbox-deploy").show();
	} else if(modus === "run"){
		console.log("switch to run");
		$(".toolbox").hide();
		$("#menu-run").addClass("active");
		$("#toolbox-run").show();
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
	// initialize toolboxes
	// build toolbox
	$.post("/dianne/builder", {action : "available-modules"}, 
		function( data ) {
			$.each(data, function(index, name){
				console.log(name);	
				// Render toolbox item
				$('#toolbox-build').append(renderTemplate("module-build",
						{name: name }));
				
				// make draggable and add code to create new modules drag-and-drop style
				$('#'+name).draggable({helper: "clone"});
				$('#'+name).bind('dragstop', function(event, ui) {
					if(checkAddModule($(this))){
						// clone the toolbox item
					    var moduleItem = $(ui.helper).clone().removeClass("build");
					    
						addModule(moduleItem, $(this));
					}
				});
			});
		}
		, "json");
	
	// show correct mode
	setModus(modus);
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
	var dialog;
	dialog = $("#"+dialogId);
	if(dialog.length==0){
		// create new dialog
		var d = renderTemplate("dialog", {
			id : id,
			title : "Configure module"
		});
		dialog = $(d);
	}
	
	// TODO check which "mode" you are in, for now only "build" mode
	if(modus==="build"){
		dialog = createBuildModuleDialog(id, dialog);
	} else if(modus==="deploy"){
		dialog = createDeployModuleDialog(id, dialog);
	}
	
	var offset = moduleItem.offset();
	offset.top = offset.top - 100;
	offset.left = offset.left - 200;
	
	// show the modal (disable backdrop)
	dialog.modal({'show':true, 'backdrop':false}).draggable({handle: ".modal-header"}).offset(offset);
	
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
				console.log(property);	
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

/*
 * Module/Connection add/remove checks
 */

/**
 * Check whether one is allowed to instantiate another item from this tooblox
 */
function checkAddModule(toolboxItem){
	if(modus!=="build"){
		return false;
	}
	return true;
}

/**
 * Check whether one is allowed to remove this module
 */
function checkRemoveModule(moduleItem){
	if(modus!=="build"){
		return false;
	}
	return true;
}

/**
 * Check whether one is allowed to instantiate this connection
 */
function checkAddConnection(connection){
	if(modus!=="build"){
		return false;
	}
	if(deployment[connection.sourceId]!==undefined
		|| deployment[connection.targetId]!==undefined){
		return false;
	}
	return true;
}

/**
 * Check whether one is allowed to remove this connection
 */
function checkRemoveConnection(connection){
	if(modus!=="build"){
		return false;
	}
	if(deployment[connection.sourceId]!==undefined
		|| deployment[connection.targetId]!==undefined){
		return false;
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
    offset.left = moduleItem.offset().left - ($("#canvas").offset().left - $(".toolbox").offset().left);
    offset.top = moduleItem.offset().top - ($("#canvas").offset().top - $(".toolbox").offset().top);
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