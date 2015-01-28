
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
					
		// make all the window divs draggable (intially canvas will be empty so not needed...)						
		// jsPlumb.draggable($(".module.draggable"), { grid: [20, 20] });		
	
		//
		// listen for connection add/removes
		//
		jsPlumb.bind("beforeDetach", function(connection) {
			console.log("connection detach " + connection.sourceId + " -> " + connection.targetId);
			// TODO check whether connection can be detached
			return true;
		});
		
		jsPlumb.bind("beforeDrop", function(connection) {
			console.log("connection add " + connection.sourceId + " -> " + connection.targetId);
			// TODO check whether connection is OK?
			return true;
		});
	});

});

$( ".module.toolbox" ).click(function() {
	var module = $( this ).clone().removeClass("toolbox").addClass("draggable").appendTo("#canvas");
	// TODO configure module dialog?
	var type = module.attr("id");
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
	
	jsPlumb.draggable(module);
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
