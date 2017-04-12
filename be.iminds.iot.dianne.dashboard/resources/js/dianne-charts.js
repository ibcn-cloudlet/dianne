

// pie chart for the system status
function createStatusChart(container, status){
	container.highcharts({
        chart: {
            plotBackgroundColor: null,
            plotBorderWidth: null,
            plotShadow: false,
            height: 220,
            type: 'pie'
        },
        legend: {
            layout: 'horizontal',
            floating: true,
            x: -10,
            y: 20
        },
        credits: {
            enabled: false
        },
        title:{
            text:''
        },
        tooltip: {
            pointFormat: '{series.name}: <b>{point.y}</b>'
        },
        plotOptions: {
            pie: {
                allowPointSelect: true,
                cursor: 'pointer',
                dataLabels: {
                    enabled: false
                },
                showInLegend: true
            }
        },
        series: [{
            name: 'Utilization',
            colorByPoint: true,
            data: [{
                name: 'Learning',
                y: status.learn
            }, {
                name: 'Idle',
                y: status.idle
            }, {
                name: 'Evaluating',
                y: status.eval
            }, {
                name: 'Acting',
                y: status.act
            }]
        }]
    });	
}


// create job results chart
function createResultChart(container, job, scale){
	// in case of learn chart, plot learn progress curve
	if(job.type==="LEARN"){
		DIANNE.learnResult(job.id).then(function(learnprogress){
			if(learnprogress[0] === undefined){
				return;
			}
			
			var series = [];
			$.each(learnprogress, function(i) {
				var progress = learnprogress[i];
				var s = 0;
				for (var key in progress) {
					if (progress.hasOwnProperty(key) && key !=='iteration') {
						var ok = false;
						// add data point to correct series
						$.each(series, function(i) {
							if(series[i].name === key){
								series[i].data.push({x: progress.iteration, y:progress[key]});
								ok = true;
							}
						});
						if(!ok){
							// add new series
							var serie = {name: key, data: []};
							serie.data.push({x: progress.iteration, y:progress[key]});
							series.push(serie);
						}
						
					}
				}
			});
			
			createLineChart(container, 'Iterations', 'Learn progress', scale, series);
		});
	} else if(job.type==="EVALUATE"){
		// in case of evaluate jobs, plot a progress bar if still busy, confusion matrix heatmap otherwise
		container.empty();
		DIANNE.evaluationResult(job.id).then(function(evaluations){
			$.each(evaluations, function(i) {
				var eval = evaluations[i];
				if(eval.confusionMatrix!==undefined){
					// TODO what with multiple evaluations?
					if(eval.confusionMatrix.length <= 20)
						createConfusionChart(container, eval.confusionMatrix, scale);
					else {
						container.prepend( "<div><b>Top-5 accuracy:</b> "+eval.top5*100+" %</div><br/>" );
						container.prepend( "<div><b>Top-3 accuracy:</b> "+eval.top3*100+" %</div><br/>" );
					}
					container.prepend( "<div><b>Accuracy:</b> "+eval.accuracy*100+" %</div><br/>" );

				} else {				
					if(eval.processed!==undefined){
						createProgressBar(container, 100*eval.processed/eval.total, eval.processed+"/"+eval.total+" samples processed");
					} else if(eval.error !== undefined){
						container.prepend( "<div><b>Error:</b> "+eval.error+"</div><br/>" );
					}
				}
			});
		});
	} else if(job.type==="ACT"){
		container.empty();
 	  	DIANNE.agentResult(job.id).then(function(results){
 	  		var series = [];
			$.each(results, function(i) {
				var agentprogress = results[i];
				var reward = [];
				
				$.each(agentprogress, function(k) {
					var progress = agentprogress[k];
					reward.push({
						x: progress.sequence,
		                y: progress.reward
		            });
				});
				
				series.push({name:'reward', data: reward});
			});
			
			createLineChart(container, 'Sequences', 'Reward', scale, series);
		});
	}
}

function updateResultsChart(container, data){
	var index = Number(container.attr("data-highcharts-chart"));
	if(isNaN(index))
		return;
	
	var x;
	var y;
	
	for (var key in data) {
		if (data.hasOwnProperty(key) 
				&& key !=='iteration'
				&& key !=='sequence'	
				&& key !=='worker'
				&& key !=='jobId'
				&& key !=='type') {
			// get data point (x might be iteration (learn progress) or sequence (agent progress))
			if(data.iteration !== undefined){
				x = Number(data.iteration);
			}else if(data.sequence !== undefined){
				x = Number(data.sequence);
			}
			y = Number(data[key]);

			// check out which series 
			var s = 0;
			$.each(Highcharts.charts[index].series, function(i) {
				if(Highcharts.charts[index].series[i].name === key){
					s = i;
				}
			});
			if(data.worker !== undefined){
				s = Number(data.worker);
			}
			
			if(Highcharts.charts[index].series[s]!==undefined){
				Highcharts.charts[index].series[s].addPoint([x, y], true, false, false);
			}
		}
	}

}


// generic line chart
function createLineChart(container, xAxis, yAxis, scale, series) {
	if(scale === undefined){
		scale = 1;
	}

	// adjust chart width and height to dialog
	var w = 500*scale;
	var h = 250*scale;
	
	var maxW = $('.modal-dialog').width()+10;
	if(maxW == 10){
		maxW = $('#dashboard').width()+20;
	}
	
	while(w > maxW){
		w = w/2;
		h = h/2;
	}
	
    container.highcharts({
        chart: {
            type: 'line',
            animation: false, // don't animate in old IE
            marginRight: 10,
    		height: h,
    		width: w
        },
        title : {
        	text: null
        },
        xAxis: {
            tickPixelInterval: 150,
            title: {
                text: xAxis
            },
        },
        plotOptions:{
            series:{
                turboThreshold: 1000000
            }
        },
        yAxis: {
            title: {
                text: yAxis
            },
            plotLines: [{
                value: 0,
                width: 1,
                color: '#808080'
            }],
            softMin: 0,
            softMax: 0
        },
        legend:{
            layout:'vertical',
            align:'right',
            verticalAlign:'top',
            backgroundColor:'#fff',
            borderColor:'#ccc',
            borderWidth:.5,
            y:30,
            x:0,
            itemWidth:135,
            itemStyle:{
                fontWeight:'bold'
            },
            itemHiddenStyle:{
                fontWeight:'bold'
            }
        },
        exporting: {
            enabled: false
        },
        credits: {
            enabled: false
        },
        series: series
    });
}

// confusion matrix heatmap chart
function createConfusionChart(container, matrix, scale) {
	if(scale === undefined){
		scale = 1;
	}
	// convert confusion matrix from array[x][y] to array[[x, y, val],...] 
	var data = [];
	var i,j;
	for (i = 0; i < matrix.length; i++) { 
		for (j = 0; j < matrix[i].length; j++) { 
			data.push([i, j, matrix[i][j]]);
		}
	}
	
    container.highcharts({
    	chart: {
            type: 'heatmap',
    		height: 250*scale,
    		width: 300*scale
        },
        title: {
            text: ""
        },
        credits: {
            enabled: false
        },
        colorAxis: {
            stops: [
                [0, '#3060cf'],
                [0.5, '#fffbbc'],
                [0.9, '#c4463a']
            ],
            min: 0
        },
        yAxis: {
            title: {
                text: null
            }
        }, 
        legend: {
            align: 'right',
            layout: 'vertical',
            margin: 0,
            verticalAlign: 'top'
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
            }, 
            data: data
        }]
    });
}


// progress bar
function createProgressBar(container, value, message, active){
	var progress = {};
	progress.value = value;
	if(message===undefined){
		progress.message = "";
	} else {
		progress.message = message;
	}
	if(active){
		progress.active = "progress-bar-striped active";
	} else {
		progress.active = "";
	}
	var template = $('#progress').html();
	Mustache.parse(template);
	var rendered = Mustache.render(template, progress);
	$(rendered).appendTo(container);
}


// init colors
$(function () {
	$(document).ready(function () {
		// gradient fill for pie chart
		Highcharts.getOptions().colors = Highcharts.map(Highcharts.getOptions().colors, function (color) {
	        return {
	            radialGradient: {
	                cx: 0.5,
	                cy: 0.3,
	                r: 0.7
	            },
	            stops: [
	                [0, color],
	                [1, Highcharts.Color(color).brighten(-0.3).get('rgb')] // darken
	            ]
	        };
	    });
	});
});
