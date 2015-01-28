
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
// the definition of target endpoints (will appear when the user drags a connection) 
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

jsPlumb.ready(function() {       
    // your jsPlumb related init code goes here
    console.log("init jsPlumb");
    
    jsPlumb.setContainer($("canvas"));
    jsPlumb.importDefaults({
    	ConnectionOverlays : [[ "Arrow", { location : 1 } ]],
    	Connector : [ "Flowchart", { stub:[40, 60], gap:10, cornerRadius:5, alwaysRespectStubs:true } ],
    	DragOptions : { cursor: 'pointer', zIndex:2000 },
    });
    		
	var init = function(connection) {			
		connection.getOverlay("label").setLabel(connection.sourceId.substring(15) + "-" + connection.targetId.substring(15));
		connection.bind("editCompleted", function(o) {
			if (typeof console != "undefined")
				console.log("connection edited. path is now ", o.path);
		});
	};			

	// suspend drawing and initialise.
	jsPlumb.doWhileSuspended(function() {
		
		// listen for new connections; initialise them the same way we initialise the connections at startup.
		jsPlumb.bind("connection", function(connInfo, originalEvent) { 
			init(connInfo.connection);
		});			
					
		//
		// listen for connection add/removes
		//
		jsPlumb.bind("beforeDetach", function(connection) {
			console.log("Remove connection " + connection.sourceId + " -> " + connection.targetId);
			// TODO check whether connection can be detached
			return true;
		});
		
		jsPlumb.bind("beforeDrop", function(connection) {
			console.log("Add connection " + connection.sourceId + " -> " + connection.targetId);
			// TODO check whether connection is OK?
			return true;
		});
		
		$('.toolbox').draggable({helper: "clone"});
		$('.toolbox').bind('dragstop', function(event, ui) {
		    var module = $(ui.helper).clone().removeClass("toolbox").appendTo("#canvas");
		    
		    // fix offset
		    var offset = {};
		    offset.left = module.offset().left - ($("#canvas").offset().left - $("#toolbox").offset().left);
		    offset.top = module.offset().top - ($("#canvas").offset().top - $("#toolbox").offset().top);
		    module.offset(offset);
		    
			var type = $(this).attr("id");
			var id = guid();
			module.attr("id",id);
			
			if(type==="Input"){
				jsPlumb.addEndpoint(module, source);
			} else if(type==="Output"){
				jsPlumb.addEndpoint(module, target);
			} else {
				jsPlumb.addEndpoint(module, source);
				jsPlumb.addEndpoint(module, target);
			}
			
			module.dblclick(function() {
				showConfigureModuleDialog($(this));
			});
			
			module.click(function(){
				console.log("click");
			});
			
			module.draggable(
			{
				drag: function(){
				    jsPlumb.repaintEverything();
				}
			});
			
			console.log("Add module "+id);
		});
	});

});

function showConfigureModuleDialog(module){
	var dialog = $('#configureModuleDialog');
	
	var id = module.attr("id");
	dialog.find('#configure-id').val(id);
	// set configuration options
	
	dialog.modal('show');
}

$("#configure").click(function(e){
	// apply configuration
	var data = $('#configureModuleDialog').find('form').serializeArray();
	
	$('#configureModuleDialog').modal('hide');
});

$("#delete").click(function(e){
	// remove object
	var id = $('#configureModuleDialog').find('#configure-id').val();
	
	console.log("Remove module "+id);
	
	var module = $('#'+id);
	// delete this module
	$.each(jsPlumb.getEndpoints(module), function(index, endpoint){
		jsPlumb.deleteEndpoint(endpoint)}
	);
	
	jsPlumb.detachAllConnections(module);
	module.remove();
	
	$('#configureModuleDialog').modal('hide');
});

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
