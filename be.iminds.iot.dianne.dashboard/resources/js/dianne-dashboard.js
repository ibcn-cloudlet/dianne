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

var currentMode;

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
    } else if(job.type==="ACT"){
    	DIANNE.act(job.nn, job.dataset, job.config);
    }
}

function resubmitJob(jobId){
	DIANNE.job(jobId).then(function(job){
		if(clean!==undefined){
			job.config['clean'] = $('#clean').is(':checked');
		}
		
	    if(job.type==="LEARN"){
	    	DIANNE.learn(job.nn, job.dataset, job.config);
	    } else if(job.type==="EVALUATE"){
	    	DIANNE.eval(job.nn, job.dataset, job.config);
	    } else if(job.type==="ACT"){
	    	DIANNE.act(job.nn, job.dataset, job.config);
	    }
	});
}

function stopJob(jobId){
	DIANNE.stop(jobId);
}

function refreshJobs(){
 	// queued jobs
 	DIANNE.queuedJobs().then(function(data){
 		$("#jobs-queue").empty();
 	    $.each(data, function(i) {
 	        var job = data[i];
 	        var template = $('#job-item').html();
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
     	  	job.devices = job.targets.length;
     	  	job.time = moment(job.started).from(moment());
 	        
     	  	// for dashboard list
 	        var template = $('#job-item').html();
     	  	Mustache.parse(template);
     	  	var rendered = Mustache.render(template, job);
     	  	$(rendered).appendTo($("#jobs-running"));
     	  	

     	  	
     	  	// if new running job, add jobs details (hidden)
     	  	if(!$("#"+job.id).length){
	     	  	var template2 = $('#job').html();
	     	  	Mustache.parse(template2);
	     	  	var rendered2 = Mustache.render(template2, job);
	     	  	var panel = $(rendered2).prependTo($("#dashboard"))
	     	  	if(currentMode!=="jobs"){
	     	  		panel.hide();
	     	  	}
	     	  	
	     	  	// initialize error/q chart
	     	  	createResultChart($('#'+job.id+"-progress"), job);
     	  	}
     	  	
     	  	if(job.type==="EVALUATE" || job.type==="ACT"){
	     	  	// update progress bars
	     	  	createResultChart($('#'+job.id+"-progress"), job);
	     	 }
 	    });
 	});
 	
 	// finished jobs
 	DIANNE.finishedJobs().then(function(data){
 	 	$("#jobs-finished").empty();
 	    $.each(data, function(i) {
 	        var job = data[i];
 	        var template = $('#job-item').html();
     	  	Mustache.parse(template);
     	  	var rendered = Mustache.render(template, job);
     	  	$(rendered).prependTo($("#jobs-finished"));
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
	notification.time = moment(notification.timestamp).fromNow();
	notification.level = notification.level.toLowerCase();
	var template = $('#notification').html();
	Mustache.parse(template);
	var rendered = Mustache.render(template, notification);
	$(rendered).prependTo($("#alerts"));
}



function showDetails(jobId){
	DIANNE.job(jobId).then(function(job){
		job.submitTime = moment(job.submitted).format("hh:mm:ss YYYY:MM:DD");
		if(job.started === 0){
			job.startTime = "N/A"
		} else {
			job.startTime = moment(job.started).format("hh:mm:ss YYYY:MM:DD");
		}
		if(job.stopped === 0){
			job.stopTime = "N/A"
		} else {
			job.stopTime = moment(job.stopped).format("hh:mm:ss YYYY:MM:DD");
		}
		
		var template = $('#job-details').html();
		Mustache.parse(template);
		var rendered = Mustache.render(template, job);
		var dialog = $(rendered).appendTo($("#dashboard"));
		if(job.stopped !== 0) {
			dialog.find('.cancel').hide();
			dialog.find('.resubmit').show();
		} else {
			dialog.find('.cancel').show();
			dialog.find('.resubmit').hide();
		}
		if(job.started !== 0){
			createResultChart($('#'+job.id+"-result"), job, 1.5);
		}
		dialog.on('hidden.bs.modal', function () {
			$(this).remove();
		});
		dialog.modal({
			'show' : true
		});
 	});	
}


function setModus(mode){
	currentMode = mode;
	
	$(".active").removeClass("active");
	
	if(mode === "dashboard"){
		$(".block").hide();
		$(".block").filter( ".dashboard" ).show();
		$("#mode-dashboard").addClass("active");
		
		refreshStatus();
	} else if(mode === "jobs"){
		$(".block").hide();
		refreshJobs();
		
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
     	
     	setInterval(tick, 60000);
    });
});

// here we can do any pull based stuff...
function tick(){
	// update the timestamps
	$(".timeAgo").each(function() {
		 var timestamp = Number($(this).attr("timestamp"));
		 var newTime = moment(timestamp).from(moment());
		 $(this).text(newTime);
	});
	
	$(".time").each(function() {
		 var timestamp = Number($(this).attr("timestamp"));
		 var newTime = moment.duration(moment().diff(moment(timestamp))).humanize();
		 $(this).text(newTime);
	});
}

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
	console.log("EVENT: "+event.data);
	var data = JSON.parse(event.data);
	if(data.type === "notification"){
		addNotification(data);
		refreshJobs();
		refreshStatus();
		
 	    // update final graphs in running job overview
		if(data.jobId!==undefined && data.level==="success"){
			DIANNE.job(data.jobId).then(function(job){
	     	  	createResultChart($('#'+job.id+"-progress"), job);
	 	    });
		}
	} else if(data.type === "progress"){
		var index = Number($("#"+data.jobId+"-progress").attr("data-highcharts-chart"));
    	var x = Number(data.iteration);
    	var y;
    	if(data.q!==undefined){
    		y = Number(data.q);
    	} else {
    		y = Number(data.miniBatchError);
    	}
		Highcharts.charts[index].series[0].addPoint([x, y], true, true, false);
		if(data.validationError !== undefined){
			var v = Number(data.validationError);
			Highcharts.charts[index].series[1].addPoint([x, v], true, true, false);
		}
	}
}


/**
 * Redirect to builder if exists
 */
function redirect(){
	$.ajax({
	    type: 'HEAD',
	    url: '../builder/builder.html',
	    	success: function() {
	    		window.location = '../builder/builder.html'
	    	},
	    	error: function() {
	    	}
	});
}
