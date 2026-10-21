import { makeModel, makeRng, poisson, transition, legalActions } from "./environment.js";
import { validatePolicy, chooseAction, predictionError } from "./policy.js";

const $ = id => document.getElementById(id);
const canvas = $("scene"), ctx = canvas.getContext("2d"), chart = $("deltaChart"), cctx = chart.getContext("2d");
let policy = null, model = makeModel(), rng = makeRng(404), timer = null, running = false;
let deltaHistory = [], rows = [], lastDecision = null, runSeed = 404, demand = 1, movingCars = [], fixedPhaseAge = 0;
let env = { episode_steps: 300, step_seconds: 1, arrival_rate_ns_per_step: .32,
  arrival_rate_ew_per_step: .28, discharge_per_step: 1, minimum_green_steps: 6,
  queue_bin_edges: [0, 3, 7, 12], switching_cost: 0 };

function setError(message = "") { $("errorBox").textContent = message; $("errorBox").hidden = !message; }
function queues() { return model.queues.map(q => q.length); }
function fmt(n, digits = 2) { return Number.isFinite(n) ? n.toFixed(digits) : "—"; }
function fmtTime(steps) { return `${String(Math.floor(steps / 60)).padStart(2,"0")}:${String(steps % 60).padStart(2,"0")}`; }
function clearPolicy() { policy = null; $("policyDot").classList.remove("ready"); $("policyStatus").textContent = "Baseline only · policy not loaded"; $("loadedStatus").textContent = "Not loaded"; }

async function loadPolicy(data) {
  const checked = await validatePolicy(data);
  const names = Object.keys(env);
  if (names.some(k => !(k in checked.environment)) || checked.environment.queue_bin_edges.join(",") !== checked.state_encoding.queue_bin_edges.join(","))
    throw new Error("Policy environment/state metadata is incomplete");
  policy = checked; env = { ...checked.environment };
  $("policyDot").classList.add("ready"); $("policyStatus").textContent = "Trained policy · checksum verified";
  $("loadedStatus").textContent = "Trained · loaded";
  $("metadata").textContent = JSON.stringify({algorithm: policy.algorithm, seed: policy.training.seed,
    episodes: policy.training.episodes, alpha: policy.training.learning_rate,
    gamma: policy.training.discount_factor, epsilonEnd: policy.training.epsilon_end,
    stateCount: policy.state_count, checksum: policy.sha256}, null, 2);
  $("controller").value = "rl"; setError(""); reset();
}

async function selectPolicyFile(file) {
  if (!file) return;
  try { await loadPolicy(JSON.parse(await file.text())); }
  catch (e) { clearPolicy(); setError(`Policy could not be loaded: ${e.message}`); }
}

async function tryDefaultPolicy() {
  try {
    const response = await fetch("../artifacts/policy.json");
    if (!response.ok) throw new Error("artifacts/policy.json was not found");
    await loadPolicy(await response.json());
  } catch (e) { clearPolicy(); $("metadata").textContent = "Train and export a policy, then load artifacts/policy.json. A fixed-time demo is available."; }
}

function reset() {
  pause(); model = makeModel(); runSeed = 404; rng = makeRng(runSeed); deltaHistory = []; rows = [];
  lastDecision = null; movingCars = []; fixedPhaseAge = 0; demand = Number($("demand").value); $("simClock").textContent = "00:00";
  $("runState").textContent = "Paused"; $("actionDetail").textContent = "Waiting for start";
  ["expectedReward", "observedReward", "futureValue", "deltaValue", "qValue", "stateValue", "availableActions", "chosenAction"].forEach(id => $(id).textContent = "—");
  $("deltaLabel").textContent = "—"; render();
}
function start() { if (running) return; if ($("controller").value === "rl" && !policy) { setError("The trained policy is unavailable. Load a validated policy JSON or select Fixed time."); return; }
  running = true; $("runState").textContent = "Running"; $("startBtn").disabled = true; $("pauseBtn").disabled = false;
  timer = setInterval(() => { for (let i = 0; i < Number($("speed").value); i++) tick(); }, 1000); }
function pause() { running = false; clearInterval(timer); timer = null; $("runState").textContent = "Paused"; $("startBtn").disabled = false; $("pauseBtn").disabled = true; }

function tick() {
  if (model.steps >= env.episode_steps) { pause(); $("runState").textContent = "Complete"; return; }
  const currentQueues = queues();
  const state = policy ? chooseAction(policy, model, currentQueues).state : null;
  const legal = legalActions(model.greenAge, env);
  let action;
  if ($("controller").value === "rl") action = chooseAction(policy, model, currentQueues).action;
  else action = Number(fixedPhaseAge >= Number(policy?.baseline?.phase_duration_steps || 12) && legal.length > 1);
  const qValue = policy ? policy.q_values[state][action] : NaN;
  const arrivals = [poisson(env.arrival_rate_ns_per_step * demand, rng), poisson(env.arrival_rate_ew_per_step * demand, rng)];
  const next = transition(model, action, arrivals, env);
  fixedPhaseAge = action === 1 ? 0 : fixedPhaseAge + 1;
  for (let i=0;i<next.served;i++) movingCars.push({phase:model.phase, progress:0, waitSeconds:next.servedWaits[i]});
  let td = null;
  if (policy) {
    const pred = predictionError(policy, state, action, next.reward, next.state, next.terminal, legalActions(model.greenAge, env)); td = pred.delta;
    $("expectedReward").textContent = fmt(pred.expectedReward); $("observedReward").textContent = fmt(pred.observedReward);
    $("futureValue").textContent = fmt(pred.futureContribution); $("deltaValue").textContent = `${td >= 0 ? "+" : ""}${fmt(td)}`;
    $("deltaLabel").textContent = Math.abs(td) < .05 ? "Near zero" : td > 0 ? "Positive error" : "Negative error";
    $("deltaValue").style.color = Math.abs(td) < .05 ? "#a8b8ca" : td > 0 ? "#6dc2b2" : "#efb85c";
    deltaHistory.push(td); if (deltaHistory.length > 80) deltaHistory.shift(); drawDelta();
    $("stateValue").textContent = state; $("availableActions").textContent = legal.map(a => a ? "SWITCH" : "KEEP").join(" · ");
    $("chosenAction").textContent = action ? "SWITCH" : "KEEP"; $("qValue").textContent = fmt(qValue);
  } else { $("chosenAction").textContent = action ? "SWITCH" : "KEEP"; $("availableActions").textContent = legal.map(a => a ? "SWITCH" : "KEEP").join(" · "); }
  lastDecision = { state, action, qValue, td };
  rows.push({step: model.steps, seed: runSeed, controller: $("controller").value, demand,
    queue_ns: queues()[0], queue_ew: queues()[1], arrivals_ns: arrivals[0], arrivals_ew: arrivals[1],
    reward: next.reward, td_error: td ?? "", phase: model.phase, switches: model.switches,
    completed: displayedCompleted(), waiting_seconds: model.waitingSeconds});
  $("actionDetail").textContent = `${action ? "SWITCH" : "KEEP"} · ${arrivals[0] + arrivals[1]} arrivals · ${next.served} served`;
  render();
}

function drawDelta() {
  const w = chart.width, h = chart.height; cctx.clearRect(0, 0, w, h); cctx.strokeStyle = "#293d53"; cctx.beginPath(); cctx.moveTo(0,h/2); cctx.lineTo(w,h/2); cctx.stroke();
  if (!deltaHistory.length) return;
  const scale = Math.max(1, ...deltaHistory.map(Math.abs)); cctx.strokeStyle = "#6dc2b2"; cctx.lineWidth = 2; cctx.beginPath();
  deltaHistory.forEach((v,i) => { const x = i/(Math.max(1,deltaHistory.length-1))*w, y = h/2-v/scale*(h*.42); i ? cctx.lineTo(x,y) : cctx.moveTo(x,y); }); cctx.stroke();
}

function render() {
  $("nsQueue").textContent = queues()[0]; $("ewQueue").textContent = queues()[1];
  const completed = displayedCompleted();
  const completedWait = model.completedWait - movingCars.reduce((sum, car) => sum + car.waitSeconds, 0);
  $("meanWait").textContent = fmt(completed ? completedWait/completed : 0,1);
  $("completed").textContent = completed; $("reward").textContent = fmt(model.returnTotal,1); $("switches").textContent = model.switches;
  $("simClock").textContent = fmtTime(model.steps); $("phaseAge").textContent = `Phase age ${model.greenAge} s`;
  $("phaseLabel").innerHTML = `<i class="signal ns ${model.phase===0?"active":""}"></i>${model.phase===0?"NS":"EW"} green <span class="muted">· opposing phase red</span>`;
  drawScene();
}

function drawScene() {
  const w=canvas.width,h=canvas.height,cx=w/2,cy=h/2; ctx.clearRect(0,0,w,h);
  ctx.fillStyle="#0d1b2b";ctx.fillRect(0,0,w,h); ctx.fillStyle="#283646";ctx.fillRect(cx-92,0,184,h);ctx.fillRect(0,cy-92,w,184);
  ctx.strokeStyle="#718095";ctx.setLineDash([14,13]);ctx.lineWidth=2;
  for(const [x1,y1,x2,y2] of [[cx-30,0,cx-30,cy-100],[cx+30,cy+100,cx+30,h],[0,cy-30,cx-100,cy-30],[cx+100,cy+30,w,cy+30]]){ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();}ctx.setLineDash([]);
  ctx.font="12px Segoe UI";ctx.fillStyle="#a8b8ca";ctx.textAlign="center";ctx.fillText("NORTH · NS",cx,18);ctx.fillText("SOUTH · NS",cx,h-10);ctx.fillText("WEST · EW",62,cy-108);ctx.fillText("EAST · EW",w-62,cy-108);
  // Direction lamps: only the active opposing phase is green; all other movements are red.
  const lamps=[[cx-64,cy-108,model.phase===0],[cx+64,cy+108,model.phase===0],[cx-108,cy+64,model.phase===1],[cx+108,cy-64,model.phase===1]];
  for(const [x,y,green] of lamps){ctx.beginPath();ctx.fillStyle=green?"#65c4a4":"#d26770";ctx.arc(x,y,8,0,Math.PI*2);ctx.fill();}
  const drawCars=(count,x,y,dx,dy)=>{for(let i=0;i<Math.min(count,22);i++){const xx=x+dx*i,yy=y+dy*i;ctx.fillStyle="#84a9c4";ctx.beginPath();ctx.roundRect(xx-7,yy-10,14,20,3);ctx.fill();}};
  // Queued vehicle icons stay behind the stop line. A discharge is reflected by removal from the queue.
  drawCars(Math.ceil(queues()[0]/2),cx-30,cy-110,0,-23);drawCars(Math.floor(queues()[0]/2),cx+30,cy+110,0,23);
  drawCars(Math.ceil(queues()[1]/2),cx-112,cy+30,-23,0);drawCars(Math.floor(queues()[1]/2),cx+112,cy-30,23,0);
  ctx.fillStyle="#18273a";ctx.fillRect(cx-66,cy-66,132,132);ctx.fillStyle="#a8b8ca";ctx.font="11px Segoe UI";ctx.fillText("4-WAY",cx,cy-3);ctx.fillText("INTERSECTION",cx,cy+14);
  for(const car of movingCars){const p=car.progress;ctx.fillStyle="#f0c36a";ctx.beginPath();if(car.phase===0)ctx.roundRect(cx-8,cy-12+((p-.5)*220),16,24,4);else ctx.roundRect(cx-12+((p-.5)*220),cy-8,24,16,4);ctx.fill();}
}

function displayedCompleted(){return Math.max(0,model.completed-movingCars.length);}

function animateCars(){const count=movingCars.length;movingCars.forEach(car=>car.progress+=.012);movingCars=movingCars.filter(car=>car.progress<1);if(movingCars.length!==count)render();else drawScene();requestAnimationFrame(animateCars);}

function exportCsv() {
  if (!rows.length) { setError("Run the simulation before exporting metrics."); return; }
  const cols = Object.keys(rows[0]); const csv = [cols.join(","), ...rows.map(r=>cols.map(c=>JSON.stringify(r[c])).join(","))].join("\n");
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));a.download="dopa_sync_browser_run.csv";a.click();URL.revokeObjectURL(a.href);
}

$("startBtn").addEventListener("click",start); $("pauseBtn").addEventListener("click",pause); $("resetBtn").addEventListener("click",reset);
$("csvBtn").addEventListener("click",exportCsv); $("policyFile").addEventListener("change",e=>selectPolicyFile(e.target.files[0]));
$("controller").addEventListener("change",()=>{if(running){pause();setError("Controller mode changed. Reset the simulation before resuming to keep controller state comparable.");}else reset();});
$("demand").addEventListener("change",()=>{if(running){$("demand").value=String(demand);setError("Pause and reset before changing demand; this keeps each run's demand constant.");}else reset();});
window.addEventListener("resize",drawScene); render(); requestAnimationFrame(animateCars); tryDefaultPolicy();
