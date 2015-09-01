/*
 * Javascript for the Pong game
 * 
 */

// toggle options menu
function toggle(){
	var o = document.getElementById('options');
	o.style.display = (o.style.display != 'none' ? 'none' : 'block' );
}

function start(){
	sendMessage("start");
}

function selectAI(){
	var ai = document.getElementById("ai").value;
	sendMessage("ai="+ai);
}

function selectAgent(){
	var agent = document.getElementById("agent").value;
	sendMessage("agent="+agent);
}

function selectInterval(){
	var interval = document.getElementById("interval").value;
	sendMessage("interval="+interval);
}

// score
var agentScore = 0;
var opponentScore = 0;

// websocket connection
var host = window.location.hostname;
var connection = new WebSocket('ws://'+host+':8787', 'json');
connection.onopen = function () {
	console.log('Connection Opened');
};
connection.onerror = function (error) {
	console.log('WebSocket Error ' + error);
};
connection.onmessage = function (e) {
	var state = JSON.parse(e.data);
	
	// console.log(state);
	
	if(state.x !== undefined){
		x = state.x;
		y = state.y;
		vx = state.vx;
		vy = state.vy;
		p = state.p;
		o = state.o;
	} else if(state.score !== undefined){ 
		if(state.score == -1){
			agentScore++;
		} else {
			opponentScore++;
		}
	} else {
		paddleh = state.paddleLength * HEIGHT/2;
		paddlew = state.paddleWidth * WIDTH/2;
		radius = state.ballRadius * HEIGHT/2;
		bounds = state.bounds;
	}
	draw();
};

function sendMessage(msg){
  	connection.send(msg);
}

// user control
var paction = 0; //player action
var aaction = 0; // agent action

function onKeyDown(evt) {
	var code = evt.keyCode;
	if(code == 40 || code == 76){ // down arrow or L
		// player down	
		if(paction!=1){
			paction = 1;
			sendMessage("paction="+paction);
		}
	} else if(code == 38 || code == 80){ // up arrow or P
		// player up
		if(paction!=-1){
			paction = -1;
			sendMessage("paction="+paction);
		}
	} else if(code == 65){  // A on querty
		// agent down
		if(aaction!=1){
			aaction = 1;
			sendMessage("aaction="+aaction);
		}
	} else if(code == 81){  // Q on querty
		// agent up 
		if(aaction!=-1){
			aaction = -1;
			sendMessage("aaction="+aaction);
		}
	}
}

//and unset them when the right or left key is released
function onKeyUp(evt) {
	var code = evt.keyCode;
	if(code == 40 || code == 38 || code == 80 || code == 76){
		if(paction!=0){
			paction = 0;
			sendMessage("paction="+paction);
		}
	} else if(code == 65 || code == 81){
		if(aaction!=0){
			aaction = 0;
			sendMessage("aaction="+aaction);
		}
	}
}

document.addEventListener('keydown', onKeyDown);
document.addEventListener('keyup', onKeyUp);

// pong game

// x,y pos of the ball
var x = 0;
var y = 0;

// vx,vy speed of the ball
// vx and vy can be used for interpollation in more advanced stage?
var vx = 0;
var vy = 0;

// player and opponent position
var p = 0;
var o = 0;

// width and height of the canvas
var WIDTH;
var HEIGHT;

// size of paddles
var paddleh = 100;
var paddlew = 10;
// size of ball
var radius = 10;
// field boundary
var bounds = 0.8;

// canvas
var ctx;
var canvas = document.getElementById('pongCanvas');

function init() {
	// Set canvas to biggest square
	var W = window.innerWidth * 0.8;
	var H = window.innerHeight;
	var S = W > H ? H : W;
	canvas.width = S - 35;
	canvas.height = S - 35;

	ctx = canvas.getContext("2d");
	WIDTH = canvas.width;
	HEIGHT = canvas.height;
}

function circle(x, y, r) {
	ctx.beginPath();
	ctx.arc(x, y, r, 0, Math.PI * 2, true);
	ctx.closePath();
	ctx.fill();
}

function rect(x, y, w, h) {
	ctx.beginPath();
	ctx.rect(x, y, w, h);
	ctx.closePath();
	ctx.fill();
}

function clear() {
	ctx.clearRect(0, 0, WIDTH, HEIGHT);
}

function drawSideLines() {
	ctx.beginPath();
	ctx.rect(0, HEIGHT / 2 * (1 - bounds) - 10, WIDTH, 10);
	ctx.closePath();
	ctx.fill();

	ctx.beginPath();
	ctx.rect(0, HEIGHT - HEIGHT / 2 * (1 - bounds), WIDTH, 10);
	ctx.closePath();
	ctx.fill();
}

function drawScore(){
	var size = Math.round(WIDTH / 15);
	ctx.font = size+"px Impact";
	ctx.fillText(agentScore, 0.5*size, size);
	var metrics = ctx.measureText(opponentScore);
	ctx.fillText(opponentScore, WIDTH - 0.5*size - metrics.width, size);
}

function draw() {
	clear();
	circle(x * WIDTH / 2 + WIDTH / 2, y * HEIGHT / 2 + HEIGHT / 2,
			radius);

	drawSideLines();
	drawScore();

	rect(0, p * HEIGHT / 2 + HEIGHT / 2 - paddleh / 2, paddlew,
			paddleh);
	rect(HEIGHT - paddlew, o * HEIGHT / 2 + HEIGHT / 2 - paddleh
			/ 2, paddlew, paddleh);
}

init();
draw();