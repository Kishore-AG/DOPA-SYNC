import { encodeState, legalActions } from "./environment.js";

function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.keys(value).sort().map(k => `${JSON.stringify(k)}:${canonical(value[k])}`).join(",")}}`;
  return JSON.stringify(value);
}

export async function validatePolicy(data) {
  if (!data || data.schema_version !== 1 || data.trained !== true || data.algorithm !== "tabular_q_learning")
    throw new Error("Unsupported or untrained policy artifact");
  if (data.actions?.["0"] !== "KEEP" || data.actions?.["1"] !== "SWITCH") throw new Error("Action mapping is invalid");
  if (!Array.isArray(data.q_values) || data.q_values.length !== data.state_count ||
      data.q_values.some(row => !Array.isArray(row) || row.length !== 2 || row.some(v => !Number.isFinite(v))))
    throw new Error("Q-table dimensions or values are invalid");
  const { sha256, ...payload } = data;
  const bytes = new TextEncoder().encode(canonical(payload));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  const actual = [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, "0")).join("");
  if (actual !== sha256) throw new Error("Policy checksum does not match");
  return data;
}

export function chooseAction(policy, model, queues) {
  const env = policy.environment;
  const state = encodeState(model.phase, queues[0], queues[1], model.greenAge, env);
  if (state < 0 || state >= policy.state_count) throw new Error("Encoded state is outside policy bounds");
  const actions = legalActions(model.greenAge, env);
  const action = actions.reduce((best, a) => policy.q_values[state][a] > policy.q_values[state][best] ? a : best, actions[0]);
  return { state, action, q: policy.q_values[state][action], actions };
}

export function predictionError(policy, state, action, reward, nextState, terminal, nextLegalActions = [0, 1]) {
  const future = terminal ? 0 : policy.training.discount_factor * Math.max(...nextLegalActions.map(a => policy.q_values[nextState][a]));
  return { expectedReward: policy.q_values[state][action], observedReward: reward,
    futureContribution: future,
    delta: reward + future - policy.q_values[state][action] };
}
