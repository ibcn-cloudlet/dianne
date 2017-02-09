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

// add a console for CLI input
window.addEventListener("keydown", keyPressed, false);

var consolePrefix = "<strong>g!</strong> ";
var commands = [];
var index = 0;

function keyPressed(e) {
	console.log(e.code);
	if(e.code === "F10"){
		toggleConsole();
	} else if(e.code === "Enter" || e.code === "NumpadEnter"){
		// enter
		var input = $('#console-input').html();
		var cmds = input.split("<br>");
		
		for (var i = 0; i < cmds.length; i++) {
			var command = cmds[i];
			
			if(command.length === 0 || command.startsWith("#")){
				continue;
			}
			
			// process command
			commands.push(command);
			index = commands.length;
			var output = $('#console-output').html();
			output += "<br/>"+consolePrefix+command;
			$('#console-output').html(output);
			$('#console').scrollTop($('#console')[0].scrollHeight);
			
			$.post("/dianne/console", {'command':command}, 
					function( data ) {
						var output = $('#console-output').html();
						output += "<br/>"+data;
						$('#console-output').html(output.replace(new RegExp('\r?\n','g'), '<br>'));
						$('#console').scrollTop($('#console')[0].scrollHeight);
					}
					, "text");
		}
		
		$('#console-input').text("");
		$('#console-input').focus();
		e.preventDefault();
	} else if(e.code === "ArrowUp"){
		if(index > 0){
			index--;
			var command = commands[index];
			$('#console-input').text(command);
		} else {
			$('#console-input').text("");
		}
	} else if(e.code === "ArrowDown"){
		if(index < commands.length){
			index++;
			var command = commands[index];
			$('#console-input').text(command);
		} else {
			$('#console-input').text("");
		}
	}
}

function toggleConsole(){
	if($('#console').hasClass('hidden')){
		$('#console').removeClass('hidden')
		$('#console-input').focus();
		$('#console-toggle').html('&#652;');
	} else {
		$('#console').addClass('hidden');
		$('#console-toggle').html('v');
	}
}
