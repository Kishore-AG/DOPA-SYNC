export const KEEP = 0, SWITCH = 1;

export function queueBin(length, edges) {
  let bin = 0;
  for (const edge of edges.slice(1)) if (length >= edge) bin++;
  return bin;
}

export function encodeState(phase, nsQueue, ewQueue, greenAge, env) {
  const count = env.queue_bin_edges.length;
  const age = Math.min(greenAge, env.minimum_green_steps);
  return (((phase * count + queueBin(nsQueue, env.queue_bin_edges)) * count +
    queueBin(ewQueue, env.queue_bin_edges)) * (env.minimum_green_steps + 1) + age);
}

export function legalActions(greenAge, env) {
  return greenAge >= env.minimum_green_steps ? [KEEP, SWITCH] : [KEEP];
}

export function transition(model, action, arrivals, env) {
  if (![KEEP, SWITCH].includes(action)) throw new Error("Invalid action");
  if (action === SWITCH && model.greenAge < env.minimum_green_steps) throw new Error("Switch before minimum green");
  if (action === SWITCH) { model.phase = 1 - model.phase; model.greenAge = 0; model.switches++; }
  model.queues.forEach((q, i) => { for (let n = 0; n < arrivals[i]; n++) q.push(0); });
  const waiting = (model.queues[0].length + model.queues[1].length) * env.step_seconds;
  model.queues.forEach(q => { for (let i = 0; i < q.length; i++) q[i] += env.step_seconds; });
  let served = 0, servedWaits = [];
  for (let n = 0; n < env.discharge_per_step && model.queues[model.phase].length; n++) {
    const age = model.queues[model.phase].shift(); model.completedWait += age; servedWaits.push(age); served++; model.completed++;
  }
  model.steps++; model.greenAge = Math.min(env.minimum_green_steps, model.greenAge + 1);
  const qTotal = model.queues[0].length + model.queues[1].length;
  const reward = -qTotal - (action === SWITCH ? env.switching_cost : 0);
  model.waitingSeconds += waiting; model.returnTotal += reward;
  return { state: encodeState(model.phase, ...model.queues.map(q => q.length), model.greenAge, env),
    reward, terminal: model.steps >= env.episode_steps, served, servedWaits, waiting };
}

export function makeModel() {
  return { phase: 0, greenAge: 0, queues: [[], []], steps: 0, waitingSeconds: 0,
    completed: 0, completedWait: 0, switches: 0, returnTotal: 0 };
}

export function makeRng(seed) {
  let state = seed >>> 0;
  return () => { state = (1664525 * state + 1013904223) >>> 0; return state / 4294967296; };
}

export function poisson(lambda, rng) {
  const limit = Math.exp(-lambda); let k = 0, product = 1;
  do { k++; product *= rng(); } while (product > limit);
  return k - 1;
}
