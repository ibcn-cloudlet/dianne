/**
 * This script allows to create a NN structure by drag-and-drop using jsPlumb
 */

// keep a map of neural network modules
var nn = {};
// keep a map of all learn blocks
var learn = {};
// keep a map of all run blocks
var run = {};

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
	// hide all modals
	$(".modal").modal('hide');
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
	
	
	// add to one of the module maps
	if(mode==="build"){
		nn[id] = module;
	} else if(mode==="learn"){
		learn[id] = module;
	} else if(mode==="run"){
		run[id] = module;
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
		delete learn[id];
	} else if(mode==="run"){
		delete run[id];
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
		if(learn[connection.sourceId]!==undefined){
			learn[connection.sourceId].input = connection.targetId; 
		} else {
			run[connection.sourceId].input = connection.targetId; 
		}
	} else if(nn[connection.targetId]===undefined){
		if(learn[connection.targetId]!==undefined){
			learn[connection.targetId].output = connection.sourceId; 
		} else {
			run[connection.targetId].output = connection.sourceId; 
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
		if(learn[connection.sourceId]!==undefined){
			delete learn[connection.sourceId].input; 
		} else {
			delete run[connection.sourceId].input; 
		}
	} else if(nn[connection.targetId]===undefined){
		if(learn[connection.targetId]!==undefined){
			delete learn[connection.targetId].output; 
		} else {
			delete run[connection.targetId].output; 
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
 * Module configuration/deletion dialog stuff 
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
			dialog = createDeployModuleDialog(id, dialog);
		} else if (currentMode === "learn") {
			dialog = createLearnModuleDialog(id, dialog);
		} else if (currentMode === "run") {
			dialog = createRunModuleDialog(id, dialog);
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
	
	dialog.find('.content').append("<br/>");
	
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
					nn[item.name] = item.value;
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


function createDeployModuleDialog(id, dialog){
	var module = nn[id];
	
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
	var block = other[id];
	if(block===undefined){
		return undefined; // no dialogs for build modules
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
		dialog.find(".exec").remove();
		
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
		
		dialog.find(".exec").click(function(e){
			var id = $(this).closest(".modal").find(".module-id").val();
			
			var trainer = other[id];
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
		
		dialog.find(".exec").click(function(e){
			var id = $(this).closest(".modal").find(".module-id").val();

			evaluate(id);
		});
	}
	
	return dialog;
}


function createRunModuleDialog(id, dialog){
	var block = other[id];
	if(block===undefined){
		return undefined; // no dialogs for build modules
	}
	
	var body = renderTemplate("dialog-body-run", {
		id : block.id,
		type : block.type
	});
	dialog.find(".modal-body").empty();
	dialog.find(".modal-body").append(body);
	
	var buttons = renderTemplate("dialog-buttons-run", {});
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
	
	
	if(block.type==="CanvasInput"){
		dialog.find(".modal-title").text("Draw your input");

		dialog.find(".run-canvas").append("<canvas class='inputCanvas' width='224' height='224' style=\"border:1px solid #000000; margin-left:150px\"></canvas>");
		dialog.find(".run-canvas").append("<button class='btn' onclick='clearCanvas()' style=\"margin-left:10px\">Clear</button>");
		
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
		
		
	} else if(block.type==="DatasetInput"){
		dialog.find(".modal-title").text("Input a sample of the MNIST dataset");

		dialog.find(".run-canvas").append("<canvas class='sampleCanvas' width='224' height='224' style=\"border:1px solid #000000; margin-left:150px\"></canvas>");
		dialog.find(".run-canvas").append("<button class='btn' onclick='sample()' style=\"margin-left:10px\">Sample</button>");
		
		sampleCanvas = dialog.find('.sampleCanvas')[0];
		sampleCanvasCtx = sampleCanvas.getContext('2d');
		
	} else if(block.type==="ProbabilityOutput"){

		dialog.find(".modal-title").text("Output probabilities");

		createOutputChart(dialog.find(".run-canvas"));
		eventsource = new EventSource("run");
		eventsource.onmessage = function(event){
			var data = JSON.parse(event.data);
			var index = Number($("#dialog-"+id).find(".run-canvas").attr("data-highcharts-chart"));
			Highcharts.charts[index].series[0].setData(data, true, true, true);
		};
		
		dialog.on('hidden.bs.modal', function () {
		    eventsource.close();
		});
	}
	
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

	eventsource = new EventSource("learner");
	eventsource.onmessage = function(event){
		var data = JSON.parse(event.data);
		var index = Number($("#dialog-"+id).find(".error").attr("data-highcharts-chart"));
    	var x = Number(data.sample);
        var y = Number(data.error); 
		Highcharts.charts[index].series[0].addPoint([x, y], true, true, false);
	};
	$.post("/dianne/learner", {"action":"learn",
		"config":JSON.stringify(other),
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
	var index = Number($("#dialog-"+id).find(".evaluate").attr("data-highcharts-chart"));
	Highcharts.charts[index].series[0].setData(null, true, true, false);
	$("#dialog-"+id).find(".accuracy").text("");

	eventsource = new EventSource("learner");
	eventsource.onmessage = function(event){
		var data = JSON.parse(event.data);
		var index = Number($("#dialog-"+id).find(".evaluate").attr("data-highcharts-chart"));
		Highcharts.charts[index].series[0].setData(data, true, true, false);
	};
	$.post("/dianne/learner", {"action":"evaluate",
		"config":JSON.stringify(other),
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

/*
 * Save and load 
 */

function save(){
	console.log("save");
	// save modules
	var modulesJson = JSON.stringify(nn);
	
	// save layout
	var layout = saveLayout();
    var layoutJson = JSON.stringify(layout);
    console.log(layoutJson);
    
	$.post("/dianne/save", {"modules":modulesJson, "layout":layoutJson}, 
		function( data ) {
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

function load(){
	console.log("load");
	
	$.post("/dianne/load", {}, 
			function( data ) {
				nn = data.modules;
				loadLayout(data.layout);
		
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
	var moduleBlock = renderTemplate("module",
		{name: module.type, type: id, clazz: "build" }); // this is not intuitive...
	$("#canvas").append(moduleBlock);
	$('#'+id).draggable();
	$('#'+id).css('position','absolute');
	$('#'+id).css('left', posX);
	$('#'+id).css('top', posY);
	
	setupModule($('#'+id), module.type);
	jsPlumb.repaint(id);
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

function renderTemplate(template, options, target){
	var template = $('#'+template).html();
	Mustache.parse(template);
	var rendered = Mustache.render(template, options);
	return $(rendered).appendTo(target);
}
