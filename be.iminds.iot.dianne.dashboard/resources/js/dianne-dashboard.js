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
    		if(this.value !== undefined && this.value !==""){
    			job.config = configStringToObject(this.value);
    		} else {
    			job.config = {};
    		}
    	} else {
    		job[this.name] = this.value || '';
    	}
    });
    
    // check if it is an upload nn
    if(uploadNN !== undefined && job.nn == uploadNN.name){
    	job.nn = uploadNN;
    }
    
    // insert name attribute to the config
    if(job.name!==undefined && job.name!==""){
    	job.config.name = job.name;
	}

    // insert custom strategy implementation to the config
    if(job.strategy!==undefined && job.strategy!==""){
    	job.config.strategy = job.strategy;
	}
    
    if(job.type==="LEARN"){
    	DIANNE.learn(job.nn, job.dataset, job.config);
    } else if(job.type==="EVALUATE"){
    	DIANNE.eval(job.nn, job.dataset, job.config);
    } else if(job.type==="ACT"){
    	DIANNE.act(job.nn, job.dataset, job.config);
    }
}

// also allow to upload a nn modules.txt or strategy .java file from your filesystem
var uploadNN = undefined;
function upload(evt){
    var f = evt.target.files[0]; 
    if (f) {
    	var r = new FileReader();
    	r.onload = function(e) { 
    		var contents = e.target.result;
    		
    		if(f.name==="modules.txt"){
    			// nn uploaded
	    		uploadNN = JSON.parse(contents);
				$('#submit-nn').val(uploadNN.name);
    		} else if(f.name.endsWith(".java")){
    			// strategy uploaded
    			$('#submit-strategy').val(contents);
    		}
    	}
    	r.readAsText(f);
    } else {
    	alert("You should upload a modules.txt file defining the neural network.");
    }
}
document.getElementById('nn-file-input').addEventListener('change', upload, false);
document.getElementById('strategy-file-input').addEventListener('change', upload, false);


function resubmitJob(jobId){
	DIANNE.job(jobId).then(function(job){
		if(clean!==undefined && job.type==="LEARN"){
			job.config['clean'] = $('#clean').is(':checked');
		}
		
		$("#submit-name").val(job.name);
		$("#submit-type").val(job.type);
		$("#submit-nn").val(job.nn);
		$("#submit-dataset").val(job.dataset);
		$("#submit-config").val(configObjectToString(job.config));
		$("#submit-strategy").val(job.config.strategy);
		
		$('#submit-modal').modal('show');
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

 	 	$("#status").empty();
 	    var template = $('#stat').html();
 	    Mustache.parse(template);
     	var rendered = Mustache.render(template, status);
     	$(rendered).appendTo($("#status"));
 	   
		// status chart
        createStatusChart($('#status-chart'), status);
 	});
}

function addNotification(notification){
	notification.time = moment(Number(notification.timestamp)).fromNow();
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
		job.tag = job.config.tag;
		job.config = configObjectToString(job.config);
		
		var template = $('#job-details').html();
		Mustache.parse(template);
		var rendered = Mustache.render(template, job);
		var dialog = $(rendered).appendTo($("#dashboard"));
		if(job.stopped !== 0) {
			dialog.find('.cancel').hide();
			dialog.find('.resubmit').show();
			
			if(job.type !== "LEARN"){
				dialog.find('.clean').hide();
			} else {
				dialog.find('.clean').show();
			}
			
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


// initialize
$(function () {
    $(document).ready(function () {
		refreshStatus();

     	// TODO set each time the dialog is shown?
     	// nn options in submission dialog
     	DIANNE.nns().then(function(data){
     		$("#submit-nn").typeahead({
     			source : data,
     		    updater: function (item) {
     		        var terms = split(this.query);
     		        terms.pop();
     		        terms.push(item);
     		        return terms.join(", ");
     		    },
     		    matcher: function (item) {
     		    	var q = extractLast(this.query);
     		    	return (item.toLowerCase().indexOf(q.toLowerCase()) >= 0);
     		    },
     		    highlighter: function (item) {
     		        var query = extractLast(this.query).replace(/[\-\[\]{}()*+?.,\\\^$|#\s]/g, '\\$&');
     		        return item.replace(new RegExp('(' + query + ')', 'ig'), function ($1, match) {
     		            return '<strong>' + match + '</strong>';
     		        });
     		    }
     		});
     		
     		
     	});
     	// dataset options in submission dialog
     	DIANNE.datasets().then(function(data){
     		var options = $("#submit-dataset");
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
     	
     	// allow to pass a jobId as query string for immediate details display 
    	var queryString = window.location.search;
    	if(queryString !== ""){
    		showDetails(queryString.substring(1));
    	}
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
	console.log("EVENT: "+event.data);
	var data = JSON.parse(event.data);
	if(data.type === "notification"){
		addNotification(data);
		refreshStatus();
		refreshJobs();
	} else if(data.type === "progress"){
		updateResultsChart($("#"+data.jobId+"-result"), data);
	}
}

//here we can do any pull based stuff...
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

/**
 * Config string to object conversions
 */

function configStringToObject(string){
	var configArray = string.trim().split(' ');
	var configJson = {};
	$.each(configArray, function(){
		var n = this.indexOf("=");
		configJson[this.substr(0, n)] = this.substr(n+1);
	});
	return configJson;
}

function configObjectToString(object){
	var configString = "";
	$.each(object, function(k, v) {
		if(v.length > 50){
			 v = v.substr(0, 50) + "\u2026";
		}
		if(k !== "name" && k !== "strategy") // exclude name and strategy here
			configString += k + "=" + v + " ";
	});
	return configString;
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

/**
 * Autocomplete for submit-nn
 */
var split = function (val) {
    return val.split(/\s*[,;]\s*/);
};

var extractLast = function (term) {
    return split(term).pop();
};