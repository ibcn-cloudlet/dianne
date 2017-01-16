

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
			if(learnprogress[0] !== undefined && learnprogress[0].q !== undefined){
				 var q = [];
				 $.each(learnprogress, function(i) {
					var progress = learnprogress[i];
					q.push({
						x: progress.iteration,
		                y: progress.q
		            });
				 });
				 createQChart(container, scale, q);
			} else {
				var minibatchLoss = [];
				var validationLoss = [];
				$.each(learnprogress, function(i) {
					 var progress = learnprogress[i];
					 minibatchLoss.push({
						x: progress.iteration,
		                y: progress.minibatchLoss
		             });
					 if(progress.validationLoss !== undefined){
							 validationLoss.push({
								 x: progress.iteration,
								 y: progress.validationLoss
							 });
					 }
					 
				 });
				 createLossChart(container, scale, minibatchLoss, validationLoss);	
			}
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
		// visualize act jobs as a running progress bar
		container.empty();
 	  	DIANNE.agentResult(job.id).then(function(results){
			$.each(results, function(i) {
				var agentprogress = results[i];
			
				var reward = [];
				$.each(agentprogress, function(i) {
					var progress = agentprogress[i];
					reward.push({
						x: progress.sequence,
		                y: progress.reward
		            });
				 });
				
				 createRewardChart(container, scale, reward);
			});
		});
	}
}

function updateResultsChart(container, data){
	var index = Number(container.attr("data-highcharts-chart"));
	if(isNaN(index))
		return;
	
	var x;
	var y;
	if(data.q !== undefined){
		x = Number(data.iteration);
		y = Number(data.q);
	} else if(data.minibatchLoss !== undefined){
		x = Number(data.iteration);
		y = Number(data.minibatchLoss);
	} else if(data.reward !== undefined){
		x = Number(data.sequence);
		y = Number(data.reward);
	}
	Highcharts.charts[index].series[0].addPoint([x, y], true, false, false);
	
	if(data.validationLoss !== undefined){
		var v = Number(data.validationLoss);
		Highcharts.charts[index].series[1].addPoint([x, v], true, false, false);
	}
}

function createLossChart(container, scale, minibatchLoss, validationLoss){
	createLineChart(container, 'Iterations', 'Loss', scale, 'minibatch loss', minibatchLoss, 'validation loss', validationLoss);
}

function createQChart(container, scale, q){
	createLineChart(container, 'Iterations', 'Q', scale, 'Q', q);
}

function createRewardChart(container, scale, reward){
	createLineChart(container, 'Sequences', 'Reward', scale, 'reward', reward);
}

// generic line chart
function createLineChart(container, xAxis, yAxis, scale, title, data, title2, data2) {
	// if no data specified, initialize empty
	data = initializeLineData(data);
	data2 = initializeLineData(data2);
	
	if(scale === undefined){
		scale = 1;
	}
    container.highcharts({
        chart: {
            type: 'line',
            animation: false, // don't animate in old IE
            marginRight: 10,
    		height: 250*scale,
    		width: 500*scale
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
            }]
        },
        legend: {
            enabled: false
        },
        exporting: {
            enabled: false
        },
        credits: {
            enabled: false
        },
        series: [{
            name: title,
            data: data
        },
        {
            name: title2,
            data: data2
        }]
    });
}

function initializeLineData(data){
	var i;
	if(data===undefined){
		data = [];
	    for (i = -29; i <= 0; i += 1) {
	    	data.push({
	         	x: 0,
	        	y: null
	         });
	    }
	} else if(data.length < 30){
		// if not enough data, add some empty points
		for(i = 0; i < 30-data.length; i+=1){
	    	data.unshift({
	         	x: 0,
	        	y: null
	         });
		}
	}
	return data;
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