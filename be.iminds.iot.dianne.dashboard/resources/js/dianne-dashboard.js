/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  Ghent University, iMinds
 *
 * This file is part of DIANNE.
 *
 * DIANNE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/

function submitJob(){
	var array =  $("#submit-form").serializeArray();
	
	var job = {};
    $.each(array, function() {
    	if(this.name === 'config'){
    		// parse out config to json object
    		var configArray = this.value.split(' ');
    		var configJson = {};
    		$.each(configArray, function(){
    			var split = this.split("=");
    			configJson[split[0]] = split[1];
    		});
    		job[this.name] = configJson;
    	} else {
    		job[this.name] = this.value || '';
    	}
    });
    
    // insert name attribute to the config
    if(job.name!==undefined && job.name!==""){
    	job.config.name = job.name;
	}
	
    if(job.type==="LEARN"){
    	DIANNE.learn(job.nn, job.dataset, job.config);
    } else if(job.type==="EVALUATE"){
    	DIANNE.eval(job.nn, job.dataset, job.config);
    }

}

function refreshJobs(){
 	// queued jobs
 	DIANNE.queuedJobs().then(function(data){
 		$("#jobs-queue").empty();
 	    $.each(data, function(i) {
 	        var job = data[i];
 	        var template = $('#jobs-item').html();
     	  	Mustache.parse(template);
     	  	var rendered = Mustache.render(template, job);
     	  	$(rendered).appendTo($("#jobs-queue"));
 	    });
 	});
 	
 	// running jobs
 	DIANNE.runningJobs().then(function(data){
 	 	$("#jobs-running").empty();
 	    $.each(data, function(i) {
 	        var job = data[i];
 	        var template = $('#jobs-item').html();
     	  	Mustache.parse(template);
     	  	var rendered = Mustache.render(template, job);
     	  	$(rendered).appendTo($("#jobs-running"));
 	    });
 	});
 	
 	// finished jobs
 	DIANNE.finishedJobs().then(function(data){
 	 	$("#jobs-finished").empty();
 	    $.each(data, function(i) {
 	        var job = data[i];
 	        var template = $('#jobs-item').html();
     	  	Mustache.parse(template);
     	  	var rendered = Mustache.render(template, job);
     	  	$(rendered).appendTo($("#jobs-finished"));
 	    });
 	});
}

function refreshStatus(){
	DIANNE.status().then(function(data){
		var status = data[0];
 	 	$("#status").empty();
 	    var template = $('#stat').html();
 	    Mustache.parse(template);
     	var rendered = Mustache.render(template, status);
     	$(rendered).appendTo($("#status"));
 	   
		   // status chart
        $('#status-chart').highcharts({
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
                y: 10
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
                }]
            }]
        });
 	});
}

function refreshInfrastructure(){
	DIANNE.devices().then(function(data){
 	 	$(".infrastructure").remove();
 	    $.each(data, function(i) {
 	        var device = data[i];
 	        var template = $('#device').html();
     	  	Mustache.parse(template);
     	  	var rendered = Mustache.render(template, device);
     	  	$(rendered).appendTo($("#dashboard"));
     	  	
     	  	$('#'+device.id+'-cpu').highcharts(Highcharts.merge(gaugeOptions, {
                yAxis: {
                    min: 0,
                    max: 100,
                    title: {
                        text: 'CPU usage'
                    }
                },
                series: [{
                    name: 'CPU',
                    data: [device.cpuUsage],
                    dataLabels: {
                        format: '<div style="text-align:center"><span style="font-size:25px;color:' +
                            ((Highcharts.theme && Highcharts.theme.contrastTextColor) || 'black') + '">{y:.1f}%</span></div>'
                    },
                    tooltip: {
                        valueSuffix: ' %'
                    }
                }]

            }));

            $('#'+device.id+'-mem').highcharts(Highcharts.merge(gaugeOptions, {
                yAxis: {
                    min: 0,
                    max: 100,
                    title: {
                        text: 'Memory Usage'
                    }
                },
                series: [{
                    name: 'Memory',
                    data: [device.memUsage],
                    dataLabels: {
                        format: '<div style="text-align:center"><span style="font-size:25px;color:' +
                            ((Highcharts.theme && Highcharts.theme.contrastTextColor) || 'black') + '">{y:.1f}%</span></div>'
                    },
                    tooltip: {
                        valueSuffix: ' %'
                    }
                }]
            }));
 	    });
 	});	
}

function addNotification(notification){
	var template = $('#notification').html();
	Mustache.parse(template);
	var rendered = Mustache.render(template, notification);
	$(rendered).prependTo($("#alerts"));
}

function setModus(mode){
	$(".active").removeClass("active");
	
	if(mode === "dashboard"){
		$(".block").hide();
		$(".block").filter( ".dashboard" ).show();
		$("#mode-dashboard").addClass("active");
		
		refreshStatus();

	} else if(mode === "jobs"){
		$(".block").hide();
		$(".block").filter( ".jobs" ).show();
		$("#mode-jobs").addClass("active");
		
	       // learn job charts
        $('#chart1').highcharts({
            chart: {
                height: 250,
                type: 'line'
            },
            title: {
                text: ''
            },
            xAxis: {
                categories: ['100k', '200k', '300k', '400k', '500k', '600k', '700k', '800k', '900k', '1M']
            },
            yAxis: {
            },
            credits: {
                enabled: false
            },
            legend: {
            	enabled: false
            },
            tooltip: {
                formatter: function () {
                    return '<b>' + this.series.name + '</b><br/>' +
                        this.x + ': ' + this.y;
                }
            },
            plotOptions: {
            },
            series: [{
            	name: 'Error',
                data: [2.1, 1.2, 0.8, 0.65, 0.51, 0.43, 0.31, 0.28, 0.24, 0.22]
            }]
        });
        
        $('#chart2').highcharts({
            chart: {
                height: 250,
                type: 'line'
            },
            title: {
                text: ''
            },
            xAxis: {
                categories: ['100k', '200k', '300k', '400k', '500k', '600k', '700k', '800k', '900k', '1M']
            },
            yAxis: {
            },
            credits: {
                enabled: false
            },
            legend: {
            	enabled: false
            },
            tooltip: {
                formatter: function () {
                    return '<b>' + this.series.name + '</b><br/>' +
                        this.x + ': ' + this.y;
                }
            },
            plotOptions: {
            },
            series: [{
            	name: 'Target Q',
                data: [-0.1, -0.5, -0.8, -0.7, -0.4, -0.2, 0.4]
            }]
        });
        
		
		
	} else if(mode === "infrastructure"){
     	refreshInfrastructure();
     	
		$(".block").hide();
		$(".block").filter( ".infrastructure" ).show();
		$("#mode-infrastructure").addClass("active");

	}

}



// gauge charts
var gaugeOptions = {
        chart: {
            type: 'solidgauge',
            height: 200
        },
        title: null,
        pane: {
            center: ['50%', '85%'],
            size: '140%',
            startAngle: -90,
            endAngle: 90,
            background: {
                backgroundColor: (Highcharts.theme && Highcharts.theme.background2) || '#EEE',
                innerRadius: '60%',
                outerRadius: '100%',
                shape: 'arc'
            }
        },
        tooltip: {
            enabled: false
        },
        credits: {
            enabled: false
        },
        // the value axis
        yAxis: {
        	color: {
        	    radialGradient: { cx: 0.5, cy: 0.5, r: 0.5 },
        	    stops: [
        	       [0, '#003399'],
        	       [1, '#3366AA']
        	    ]
        	},
            lineWidth: 0,
            minorTickInterval: null,
            tickPixelInterval: 400,
            tickWidth: 0,
            title: {
                y: -70
            },
            labels: {
                y: 16
            }
        },
        plotOptions: {
            solidgauge: {
                dataLabels: {
                    y: 5,
                    borderWidth: 0,
                    useHTML: true
                }
            }
        }
    };


// initialize
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
    	
     
     	setModus('dashboard');

     	
     	// TODO set each time the dialog is shown?
     	// nn options in submission dialog
     	DIANNE.nns().then(function(data){
     		var options = $("#nn");
     	    $.each(data, function(i) {
     	        options.append($("<option />").val(data[i]).text(data[i]));
     	    });
     	});
     	// dataset options in submission dialog
     	DIANNE.datasets().then(function(data){
     		var options = $("#dataset");
     	    $.each(data, function(i) {
     	        options.append($("<option />").val(data[i]).text(data[i]));
     	    });
     	});

     	DIANNE.notifications().then(function(data){
     	    $.each(data, function(i) {
     	        addNotification(data[i]);
     	    });
     	});
     	
     	refreshJobs();
    });
});


/*
 * SSE for server feedback
 */
var eventsource;

if(typeof(EventSource) === "undefined") {
	// load polyfill eventsource library
	$.getScript( "js/lib/eventsource.min.js").done(function( script, textStatus ) {
		console("Fallback to eventsource.js for SSE...");
	}).fail(function( jqxhr, settings, exception ) {
		console.log("Sorry, your browser does not support server-sent events...");
	});
} 

eventsource = new EventSource("/dianne/sse");
eventsource.onmessage = function(event){
	var notification = JSON.parse(event.data);
	addNotification(notification);
	refreshJobs();
	refreshStatus();
}

