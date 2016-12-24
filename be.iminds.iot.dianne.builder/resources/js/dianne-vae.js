


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
			if($(this).hasClass('selected')){
				// show dialog
				$("#sliders").empty();
				for(var k=0; k<means.length;k++){ 
					var slider = renderTemplate("slider", {
						i : k,
						value : means[k]
					}, $("#sliders"));
				}

				$("#latent-modal").on('shown.bs.modal', function (e) {
					renderLatent(means);
				});
				$("#latent-modal").modal();
			} else {
				// update right panel
				$('.selected').removeClass('selected');
				$(this).addClass('selected');
				var i = $(this).find('.index').val();
				$.post("/dianne/vae", {'dataset':config.dataset,'encoder':config.encoder,'sample':i,'decoder':config.decoder}, 
						function( data ) {
							$("#similar").empty();
							renderSamples($("#similar"), data);
						}
						, "json");
			}
		});
		
		var sampleCanvas = item.find('.sampleCanvas')[0];
		var sampleCanvasCtx = sampleCanvas.getContext('2d');
		render(sample.data, sampleCanvasCtx, config.type);
	});
	
}

function renderLatent(means){
	$.post("/dianne/vae", {'dataset':config.dataset,'encoder':config.encoder,'latent':JSON.stringify(means),'size':1,'decoder':config.decoder}, 
			function( samples ) {
				var sampleCanvas = $('#latent-modal').find('.sampleCanvas')[0];
				var sampleCanvasCtx = sampleCanvas.getContext('2d');
				render(samples[0].data, sampleCanvasCtx, config.type);
			}
			, "json");
}

function renderTemplate(template, options, target){
	var template = $('#'+template).html();
	Mustache.parse(template);
	var rendered = Mustache.render(template, options);
	return $(rendered).appendTo(target);
}

function sliderChanged(){
	var sliders = $("#sliders").find(".slider");
	var latents = [];
	sliders.each(function(i, slider){
		var input = $(slider).find("input")[0];
		var output = $(slider).find("output")[0];
		
		var index = $(input).attr('index');
		var value = $(input).val();
		$(output).val(value);
		
		latents.push(parseFloat(value));
	});

	renderLatent(latents);
}