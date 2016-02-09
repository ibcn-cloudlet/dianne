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
	window.alert("Submit a job!");
}

function setModus(mode){
	$(".active").removeClass("active");
	
	if(mode === "dashboard"){
		$(".block").hide();
		$(".block").filter( ".dashboard" ).show();
		$("#mode-dashboard").addClass("active");
		
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
                    y: 10
                }, {
                    name: 'Idle',
                    y: 5
                }, {
                    name: 'Evaluating',
                    y: 2
                }]
            }]
        });
        
       
        
        $('#cpu-chart').highcharts(Highcharts.merge(gaugeOptions, {
            yAxis: {
                min: 0,
                max: 100,
                title: {
                    text: 'Average CPU usage'
                }
            },
            series: [{
                name: 'CPU',
                data: [68],
                dataLabels: {
                    format: '<div style="text-align:center"><span style="font-size:25px;color:' +
                        ((Highcharts.theme && Highcharts.theme.contrastTextColor) || 'black') + '">{y:.1f}%</span></div>'
                },
                tooltip: {
                    valueSuffix: ' %'
                }
            }]

        }));

        $('#memory-chart').highcharts(Highcharts.merge(gaugeOptions, {
            yAxis: {
                min: 0,
                max: 100,
                title: {
                    text: 'Average Memory Usage'
                }
            },
            series: [{
                name: 'Memory',
                data: [54],
                dataLabels: {
                    format: '<div style="text-align:center"><span style="font-size:25px;color:' +
                        ((Highcharts.theme && Highcharts.theme.contrastTextColor) || 'black') + '">{y:.1f}%</span></div>'
                },
                tooltip: {
                    valueSuffix: ' %'
                }
            }]
        }));
        
		
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
		$(".block").hide();
		$(".block").filter( ".infrastructure" ).show();
		$("#mode-infrastructure").addClass("active");
		
        $('#gpu1-cpu').highcharts(Highcharts.merge(gaugeOptions, {
            yAxis: {
                min: 0,
                max: 100,
                title: {
                    text: 'CPU usage'
                }
            },
            series: [{
                name: 'CPU',
                data: [95],
                dataLabels: {
                    format: '<div style="text-align:center"><span style="font-size:25px;color:' +
                        ((Highcharts.theme && Highcharts.theme.contrastTextColor) || 'black') + '">{y:.1f}%</span></div>'
                },
                tooltip: {
                    valueSuffix: ' %'
                }
            }]

        }));

        $('#gpu1-mem').highcharts(Highcharts.merge(gaugeOptions, {
            yAxis: {
                min: 0,
                max: 100,
                title: {
                    text: 'Memory Usage'
                }
            },
            series: [{
                name: 'Memory',
                data: [87],
                dataLabels: {
                    format: '<div style="text-align:center"><span style="font-size:25px;color:' +
                        ((Highcharts.theme && Highcharts.theme.contrastTextColor) || 'black') + '">{y:.1f}%</span></div>'
                },
                tooltip: {
                    valueSuffix: ' %'
                }
            }]
        }));
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


// placeholder charts
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

     	
     	DIANNE.learn().then(function(data){
     		console.log("LEARNED! "+JSON.stringify(data));
     	}, function(err){
     		console.log("Error! "+err);
     	});
    });
});