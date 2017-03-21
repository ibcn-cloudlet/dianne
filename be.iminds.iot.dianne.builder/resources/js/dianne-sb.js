

var config={};
config['type'] = "laser";

var index = 0;
var state = undefined;

$( document ).ready(function() {
	var pairs = window.location.search.slice(1).split('&');
	pairs.forEach(function(pair) {
	    pair = pair.split('=');
	    config[pair[0]] = decodeURIComponent(pair[1] || '');
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
				// render
				state = result.sample;
				
				var ctx;
				
				ctx = $('#observation')[0].getContext('2d');
				render(result.observation, ctx, config.type);

				ctx = $('#state')[0].getContext('2d');
				render(result.state, ctx);

				ctx = $('#action')[0].getContext('2d');
				render(result.action, ctx);

				ctx = $('#prior')[0].getContext('2d');
				render(result.prior, ctx, 'gaussian');
				
				ctx = $('#posterior')[0].getContext('2d');
				render(result.posterior, ctx, 'gaussian');

				ctx = $('#sample')[0].getContext('2d');
				render(result.sample, ctx);

				ctx = $('#reconstruction')[0].getContext('2d');
				render(result.reconstruction, ctx, config.type);
				
				index = index + 1;
		}
		, "json");
	
}

function reset(){
	index = 0;
	state = undefined;
	update();
}