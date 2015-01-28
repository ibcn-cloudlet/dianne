
jsPlumb.ready(function() {       
    // your jsPlumb related init code goes here
    console.log("init jsPlumb");
    
    jsPlumb.setContainer($("canvas"));
    jsPlumb.importDefaults({
    	ConnectionOverlays : [[ "Arrow", { location : 1 } ]],
    	Connector : [ "Flowchart", { stub:[40, 60], gap:10, cornerRadius:5, alwaysRespectStubs:true } ],
    	DragOptions : { cursor: 'pointer', zIndex:2000 },
    });
    
	var source = {
		isSource:true,
		anchor : "Right",
//		endpoint:"Rectangle",	
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
	},		
	// the definition of target endpoints (will appear when the user drags a connection) 
	target = {
		isTarget:true,
		anchor: "Left",
//		endpoint:"Rectangle",					
		paintStyle:{ 
			fillStyle:"#333"
		},
		hoverPaintStyle:{ 
			fillStyle: "#555"
		},
//		maxConnections:-1,
	},			
	init = function(connection) {			
		connection.getOverlay("label").setLabel(connection.sourceId.substring(15) + "-" + connection.targetId.substring(15));
		connection.bind("editCompleted", function(o) {
			if (typeof console != "undefined")
				console.log("connection edited. path is now ", o.path);
		});
	};			

	// suspend drawing and initialise.
	jsPlumb.doWhileSuspended(function() {

		jsPlumb.addEndpoint("Test", source);
		jsPlumb.addEndpoint("Test2", target);
		jsPlumb.addEndpoint("Test2", source);
		
					
		// listen for new connections; initialise them the same way we initialise the connections at startup.
		jsPlumb.bind("connection", function(connInfo, originalEvent) { 
			init(connInfo.connection);
		});			
					
		// make all the window divs draggable						
		jsPlumb.draggable($(".module.draggable"), { grid: [20, 20] });		
	
		// connect a few up
//		jsPlumb.connect({uuids:["Window2BottomCenter", "Window3TopCenter"], editable:true});
        
		//
		// listen for clicks on connections, and offer to delete connections on click.
		//
		jsPlumb.bind("click", function(conn, originalEvent) {
			if (confirm("Delete connection from " + conn.sourceId + " to " + conn.targetId + "?"))
				jsPlumb.detach(conn); 
		});	
		
		jsPlumb.bind("connectionDrag", function(connection) {
			console.log("connection " + connection.id + " is being dragged. suspendedElement is ", connection.suspendedElement, " of type ", connection.suspendedElementType);
		});		
		
		jsPlumb.bind("connectionDragStop", function(connection) {
			console.log("connection " + connection.id + " was dragged");
		});

		jsPlumb.bind("connectionMoved", function(params) {
			console.log("connection " + params.connection.id + " was moved");
		});
	});

});

