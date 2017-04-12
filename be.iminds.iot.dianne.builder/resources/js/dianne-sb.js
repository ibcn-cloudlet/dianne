

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
				
				ctx = $('#observation')[0].getContext('2d');
				render(result.observation, ctx, config.type);

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
				
				ctx = $('#posterior')[0].getContext('2d');
				$('#posterior')[0].title = getTitle(result.posterior.data, true);
				gauss(result.posterior, ctx, 10);

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
				
		}
		, "json");
	
}

function reset(){
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
