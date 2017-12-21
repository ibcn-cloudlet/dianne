

var config={};
config['type'] = "laser";
config['sampleFrom'] = "posterior"
config['step'] = 1
config['limit'] = 100
config['startPrior'] = 1
config['endPrior'] = 100
config['size'] = 256

var index = 0;
var prior_state = undefined;
var posterior_state = undefined;

$( document ).ready(function() {
	var pairs = window.location.search.slice(1).split('&');
	pairs.forEach(function(pair) {
	    pair = pair.split('=');
	    config[pair[0]] = decodeURIComponent(pair[1] || '');
	});
	
	var entry = renderTemplate("entry", {size : config.size, index : index}, $("#canvas"));
	update(entry)
});

function update(entry){
	config.index = index;
	
	config.state = JSON.stringify(posterior_state);
	config.sampleFrom = "posterior"
	
	$.post("/dianne/sb", config, 
		function( result ) {
			
			if(index % config.step == 0){
				// render
				var ctx;
				
				if(result.observation !== undefined){
					ctx = entry.find('.observation')[0].getContext('2d');
					render(result.observation, ctx, config.type);
				}
	
				ctx = entry.find('.posterior')[0].getContext('2d');
				gauss(result.posterior, ctx, 10);
			
	
				if(result.posterior !== undefined){
					ctx = entry.find('.rposterior')[0].getContext('2d');
					render(result.reconstruction, ctx, config.type);
				}	
			}
			
			posterior_state = result.sample;

			config.state = JSON.stringify(prior_state);
			config.sampleFrom = "prior"

			$.post("/dianne/sb", config, 
				function( result ) {
					if(index % config.step == 0){
						// render
						var ctx;

						ctx = entry.find('.prior')[0].getContext('2d');
						gauss(result.prior, ctx, 10);
						
						if(result.prior !== undefined){
							ctx = entry.find('.rprior')[0].getContext('2d');
							render(result.reconstruction, ctx, config.type);
						}
						
						entry = renderTemplate("entry", { size: config.size, index: index + parseInt(config.step) }, $("#canvas"));
					}

					if(index >= config.startPrior)
						prior_state = result.sample;
					else 
						prior_state = posterior_state;
					
					index = index + 1;
					
					if(index < config.limit){
						update(entry);
					} else {
						entry.remove();
					}
			})			
		}
		, "json");
	
}


function renderTemplate(template, options, target){
	var template = $('#'+template).html();
	Mustache.parse(template);
	var rendered = Mustache.render(template, options);
	return $(rendered).appendTo(target);
}