

var config={};
config['type'] = "laser";

var index = 0;
var state = undefined;
var pause = true;

$( document ).ready(function() {
	var pairs = window.location.search.slice(1).split('&');
	pairs.forEach(function(pair) {
	    pair = pair.split('=');
	    config[pair[0]] = decodeURIComponent(pair[1] || '');
	});
	
    $('[data-toggle="tooltip"]').tooltip({
        content: function () {
            return this.getAttribute("title");
        },
    });
	
	index = 0;
	update();
});

function update(){
	config.index = index;
	if(state !== undefined){
		config.state = JSON.stringify(state);
	} else {
		config.state = undefined;
	}
	
	$.post("/dianne/sb", config, 
		function( result ) {
		
			index = index + 1;

			// render
			state = result.sample;
			
			var ctx;
			
			if(result.observation !== undefined){
				ctx = $('#observation')[0].getContext('2d');
				render(result.observation, ctx, config.type);
			}

			ctx = $('#state')[0].getContext('2d');
			$('#state')[0].title = getTitle(result.state.data);
			render(result.state, ctx);

			ctx = $('#action')[0].getContext('2d');
			$('#action')[0].title = getTitle(result.action.data);
			render(result.action, ctx);

			if(result.prior !== undefined){
				ctx = $('#prior')[0].getContext('2d');
				$('#prior')[0].title = getTitle(result.prior.data, true);
				gauss(result.prior, ctx, 10);
			}

			if(result.posterior !== undefined){
				ctx = $('#posterior')[0].getContext('2d');
				$('#posterior')[0].title = getTitle(result.posterior.data, true);
				gauss(result.posterior, ctx, 10);
			}

			ctx = $('#sample')[0].getContext('2d');
			$('#sample')[0].title = getTitle(result.sample.data);
			render(result.sample, ctx);

			if(result.reconstruction !== undefined){
				ctx = $('#reconstruction')[0].getContext('2d');
				render(result.reconstruction, ctx, config.type);
			}
			
			if(result.reward !== undefined){
				ctx = $('#reward')[0].getContext('2d');
				$('#reward')[0].title = getTitle(result.reward.data , true);
				gauss(result.reward, ctx, 1);
			}
			
			if(!pause && index < 100){
				update();
			}
				
		}
		, "json");
	
}

function play(){
	pause = false;
	update(true);
}

function stop(){
	pause = true;
}

function reset(){
	pause = true;
	index = 0;
	current = undefined;
	update();
}

function getTitle(data, gaussian){
	if(gaussian){
		return '<b>Mean:</b> '+data.slice(0, data.length/2).toString().replace(/[,]/g, ', ')
		+'<br/>   <b>Stdev:</b> '+data.slice(data.length/2).toString().replace(/[,]/g, ', ');
	} else {
		return data.toString().replace(/[,]/g, ', ');
	}
}


function sampleFromPrior(){
	config.sampleFrom = 'prior';
	update();
}

function sampleFromPosterior(){
	config.sampleFrom = 'posterior';
	update();
}

function sample(){
	// show dialog
	$("#sliders").empty();
	for(var k=0; k<state.data.length;k++){ 
		var slider = renderTemplate("slider", {
			i : k,
			value : state.data[k]
		}, $("#sliders"));
	}
	
	$("#sample-modal").modal();
	$("#sample-modal").on('shown.bs.modal', function() {
		sliderChanged();
    });
	
}


function sliderChanged(){
	var sliders = $("#sliders").find(".slider");
	var state = [];
	sliders.each(function(i, slider){
		var input = $(slider).find("input")[0];
		var output = $(slider).find("output")[0];
		
		var index = $(input).attr('index');
		var value = $(input).val();
		$(output).val(value);
		
		state.push(parseFloat(value));
	});

	var dims = [];
	dims.push(state.length);
	
	var sampleConfig = {};
	sampleConfig.stateSample = JSON.stringify({"dims":dims, "data":state});
	sampleConfig.decoder = config.decoder;
	
	$.post("/dianne/sb", sampleConfig, 
		function( result ) {
			if(result.reconstruction !== undefined){
				var ctx = $('#sampleReconstruction')[0].getContext('2d');
				render(result.reconstruction, ctx, config.type);
			}
	});
}


function renderTemplate(template, options, target){
	var template = $('#'+template).html();
	Mustache.parse(template);
	var rendered = Mustache.render(template, options);
	return $(rendered).appendTo(target);
}