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
    });
});