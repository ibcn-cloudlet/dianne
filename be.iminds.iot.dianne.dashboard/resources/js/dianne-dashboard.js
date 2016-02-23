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
    			var n = this.indexOf("=");
    			configJson[this.substr(0, n)] = this.substr(n+1);
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
 	 	$(".jobs").remove();

 	 	$("#jobs-running").empty();
 	    $.each(data, function(i) {
 	        var job = data[i];
     	  	job.devices = job.targets.length;
     	  	job.time = moment(job.started).from(moment());
 	        
     	  	// for dashboard list
 	        var template = $('#jobs-item').html();
     	  	Mustache.parse(template);
     	  	var rendered = Mustache.render(template, job);
     	  	$(rendered).appendTo($("#jobs-running"));
     	  	
     	  	// for jobs details
     	  	var template2 = $('#job').html();
     	  	Mustache.parse(template2);
     	  	var rendered2 = Mustache.render(template2, job);
     	  	$(rendered2).appendTo($("#dashboard"));
     	  	
     	  	if(job.type==="LEARN"){
     	  		createErrorChart($('#'+job.id+"-progress"));
     	  	}

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
		var status = data;
		// format
		status.uptime = moment.duration(moment().diff(moment(status.bootTime))).humanize();
		status.spaceLeft = status.spaceLeft/1000000000;
		status.spaceLeft = status.spaceLeft.toFixed(1);
		status.devices = status.learn+status.eval+status.idle;

 	 	$("#status").empty();
 	    var template = $('#stat').html();
 	    Mustache.parse(template);
     	var rendered = Mustache.render(template, status);
     	$(rendered).appendTo($("#status"));
 	   
		// status chart
        createStatusChart($('#status-chart'), status);
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
     	  	
     	  	createGaugeChart($('#'+device.id+'-cpu'), 'CPU usage', device.cpuUsage);
     	  	createGaugeChart($('#'+device.id+'-mem'), 'Memory usage', device.memUsage);
 	    });
 	});	
}

function addNotification(notification){
	notification.time = moment(notification.time).format("HH:mm:ss DD-MM-YYYY");
	notification.level = notification.level.toLowerCase();
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
		
	} else if(mode === "infrastructure"){
     	refreshInfrastructure();
     	
		$(".block").hide();
		$(".block").filter( ".infrastructure" ).show();
		$("#mode-infrastructure").addClass("active");
	}
}


// initialize
$(function () {
    $(document).ready(function () {
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
		console.log("Fallback to eventsource.js for SSE...");
	}).fail(function( jqxhr, settings, exception ) {
		console.log("Sorry, your browser does not support server-sent events...");
	});
} 

eventsource = new EventSource("/dianne/sse");
eventsource.onmessage = function(event){
	var data = JSON.parse(event.data);
	if(data.type === "notification"){
		addNotification(data);
		refreshJobs();
		refreshStatus();
	} else if(data.type === "progress"){
		var index = Number($("#"+data.jobId+"-progress").attr("data-highcharts-chart"));
		console.log("INDEX "+index);
    	var x = Number(data.iteration);
        var y = Number(data.error); 
		Highcharts.charts[index].series[0].addPoint([x, y], true, true, false);
	}
}

