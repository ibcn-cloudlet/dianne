


var config={};
config['type'] = "laser";

$( document ).ready(function() {
	var pairs = window.location.search.slice(1).split('&');
	pairs.forEach(function(pair) {
	    pair = pair.split('=');
	    config[pair[0]] = decodeURIComponent(pair[1] || '');
	});
	
	$.post("/dianne/vae", {'dataset':config.dataset,'encoder':config.encoder}, 
			function( data ) {
				renderSamples($("#samples"), data);
			}
			, "json");
});

function renderSamples(target, data){
	$.each(data, function(index, sample){
		
		var means = sample.latent.data.splice(0,sample.latent.data.length/2);
		var formatted = "";
		$.each( means, function( i, item ) {
			formatted = formatted+"  "+item.toFixed(3);
		});
		
		var item = renderTemplate("sample", {
			index : sample.index,
			latent : formatted
		}, target);
		
		item.tooltip();
		item.click(function(){
			$('.selected').removeClass('selected');
			$(this).addClass('selected');
			var i = $(this).find('.index').val();
			$.post("/dianne/vae", {'dataset':config.dataset,'encoder':config.encoder,'sample':i,'decoder':config.decoder}, 
					function( data ) {
						$("#similar").empty();
						renderSamples($("#similar"), data);
					}
					, "json");
		});
		
		var sampleCanvas = item.find('.sampleCanvas')[0];
		var sampleCanvasCtx = sampleCanvas.getContext('2d');
		render(sample.data, sampleCanvasCtx, config.type);
	});
	
}

function renderTemplate(template, options, target){
	var template = $('#'+template).html();
	Mustache.parse(template);
	var rendered = Mustache.render(template, options);
	return $(rendered).appendTo(target);
}