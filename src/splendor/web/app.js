"use strict";

const COLORS = ["white", "blue", "green", "red", "black"];
const ALL_TOKENS = [...COLORS, "gold"];
const COLOR_NAMES = {white: "白", blue: "蓝", green: "绿", red: "红", black: "黑", gold: "黄金"};
const PHASE_NAMES = {
  main_action: "MAIN ACTION",
  payment: "PAYMENT",
  return_tokens: "RETURN TOKENS",
  choose_noble: "CHOOSE NOBLE",
  terminal: "TERMINAL",
};

let game = null;
let busy = false;
let selectedTokens = [];

const byId = (id) => document.getElementById(id);
const html = (value) => String(value).replace(/[&<>'"]/g, (ch) => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", "'":"&#39;", '"':"&quot;"}[ch]));

function gemClass(color) { return `gem-${color}`; }

function chips(vector, className = "cost-list") {
  const entries = COLORS.filter((color) => vector[color] > 0);
  if (!entries.length) return `<span class="card-id">无需求</span>`;
  return `<div class="${className}">${entries.map((color) =>
    `<span class="cost-chip ${gemClass(color)}"><span>◆</span><b>${vector[color]}</b></span>`
  ).join("")}</div>`;
}

function actionsFor(predicate) {
  return game.legalActions.filter(predicate);
}

function miniAction(action) {
  return `<button class="mini-action" data-action-index="${action.index}" type="button">${html(action.label.replace("这张", ""))}</button>`;
}

function cardMarkup(card, options = {}) {
  const relevant = options.actions || [];
  const compact = options.compact ? " reserved-card" : "";
  return `<article class="card ${gemClass(card.bonus)}${compact}" data-card-id="${html(card.id)}" style="--card-color:var(--${card.bonus})">
    <div class="card-top"><span class="points">${card.points}</span><span class="bonus-gem ${gemClass(card.bonus)}" title="提供${COLOR_NAMES[card.bonus]}色加成"></span></div>
    <span class="card-id">${html(card.id)}</span>
    <div>${chips(card.cost)}${relevant.length ? `<div class="card-actions">${relevant.map(miniAction).join("")}</div>` : ""}</div>
  </article>`;
}

function renderMarket() {
  byId("market").innerHTML = [3, 2, 1].map((tier) => {
    const reserveDeck = actionsFor((a) => a.kind === "ReserveDeck" && a.tier === tier);
    const deck = `<div class="deck-back tier-${tier}"><span class="tier-label">TIER ${tier}</span><span class="deck-count">${game.deckCounts[tier]}</span>${reserveDeck.map((a) => `<div class="deck-action">${miniAction(a)}</div>`).join("")}</div>`;
    const cards = [...game.market[tier]].map((card) => {
      const actions = actionsFor((a) => a.cardId === card.id && (a.kind === "BuyMarket" || a.kind === "ReserveMarket"));
      return cardMarkup(card, {actions});
    });
    while (cards.length < 4) cards.push(`<div class="empty-slot">已售罄</div>`);
    return `<div class="market-row">${deck}${cards.join("")}</div>`;
  }).join("");
}

function renderSupply() {
  const tokenActions = game.legalActions.filter((action) => action.kind === "TakeDistinct" || action.kind === "TakePair");
  const stacks = ALL_TOKENS.map((color) => {
    const next = [...selectedTokens, color];
    const selectable = color !== "gold" && tokenActions.some((action) => tokenSelectionFits(action, next));
    const selectedCount = selectedTokens.filter((selected) => selected === color).length;
    return `<button class="token-stack ${gemClass(color)} ${selectedCount ? "is-selected" : ""}" data-token-color="${color}" type="button" ${selectable ? "" : "disabled"}>
      <span class="token-disc"></span><span class="token-count">${game.supply[color]}</span><span class="token-name">${COLOR_NAMES[color]}</span>
      ${selectedCount ? `<span class="selected-count">+${selectedCount}</span>` : ""}
    </button>`;
  }).join("");
  const exact = exactTokenAction();
  const selection = selectedTokens.length
    ? selectedTokens.map((color) => COLOR_NAMES[color]).join(" + ")
    : "点击宝石堆选取";
  byId("supply").innerHTML = `${stacks}<div class="token-selection">
    <span>${selection}</span>
    <div><button class="token-confirm" data-token-confirm type="button" ${exact ? "" : "disabled"}>确认拿取</button><button class="token-reset" data-token-reset type="button" ${selectedTokens.length ? "" : "disabled"}>重选</button></div>
  </div>`;
}

function tokenSelectionFits(action, selection) {
  if (action.kind === "TakePair") {
    return selection.length <= 2 && selection.every((color) => color === action.color);
  }
  if (action.kind === "TakeDistinct") {
    return selection.length <= action.colors.length
      && new Set(selection).size === selection.length
      && selection.every((color) => action.colors.includes(color));
  }
  return false;
}

function tokenSelectionExact(action, selection) {
  if (!tokenSelectionFits(action, selection)) return false;
  return action.kind === "TakePair"
    ? selection.length === 2
    : selection.length === action.colors.length;
}

function exactTokenAction() {
  return game?.legalActions.find((action) => tokenSelectionExact(action, selectedTokens));
}

function selectToken(color) {
  if (busy || !game || game.phase !== "main_action") return;
  const next = [...selectedTokens, color];
  const validPrefix = game.legalActions.some((action) => tokenSelectionFits(action, next));
  if (!validPrefix) return;
  selectedTokens = next;
  renderSupply();
  const exact = exactTokenAction();
  const canExtend = COLORS.some((nextColor) =>
    game.legalActions.some((action) => tokenSelectionFits(action, [...selectedTokens, nextColor]))
  );
  if (exact && !canExtend) performAction(exact.index);
}

function renderNobles() {
  byId("nobles").innerHTML = game.nobles.map((noble) => {
    const actions = actionsFor((a) => a.kind === "ClaimNoble" && a.nobleId === noble.id);
    return `<article class="noble" data-noble-id="${html(noble.id)}"><span class="points">${noble.points}</span>${chips(noble.requirement, "requirement-list")}${actions.map(miniAction).join("")}</article>`;
  }).join("") || `<div class="empty-slot">贵族已全部被邀请</div>`;
}

function cardSummary(card) {
  const costs = COLORS.filter((c) => card.cost[c]).map((c) => `${COLOR_NAMES[c]}${card.cost[c]}`).join(" ") || "免费";
  return `<div class="tooltip-card"><b>${COLOR_NAMES[card.bonus]}加成 · ${card.points}分</b><br><span>需求 ${costs}</span></div>`;
}

function renderReserved(entry, own) {
  if (entry.hidden) return `<div class="reserved-back tier-${entry.tier}" title="对手从 ${entry.tier} 级牌堆盲保留"></div>`;
  const actions = own && entry.card ? actionsFor((a) => a.kind === "BuyReserved" && a.cardId === entry.card.id) : [];
  return cardMarkup(entry.card, {compact: true, actions});
}

function playerPanel(player, own) {
  const groups = Object.fromEntries(COLORS.map((color) => [color, player.purchased.filter((card) => card.bonus === color)]));
  const columns = COLORS.map((color) => `<div class="asset-column">
    <div class="purchased-pile ${gemClass(color)}"><strong>${groups[color].length}</strong><small>${COLOR_NAMES[color]}色牌</small></div>
    <div class="held-token ${gemClass(color)}">${player.tokens[color]}</div>
    <div class="purchased-tooltip"><b>${COLOR_NAMES[color]}色已购牌</b>${groups[color].length ? groups[color].map(cardSummary).join("") : `<div class="tooltip-card">暂无</div>`}</div>
  </div>`).join("");
  const nobles = player.nobles.length ? `已获贵族 ${player.nobles.length} 位（${player.nobles.reduce((sum, n) => sum + n.points, 0)} 分）` : "尚未获得贵族";
  const label = own ? "你" : game.opponentLabel;
  return `<section class="player-panel ${own ? "self" : "opponent"}">
    <div class="player-head">
      <div class="player-name"><span class="avatar">${own ? "你" : "AI"}</span><span><strong>${label} ${player.isFirst ? `<em class="first-badge">先手</em>` : ""}</strong><small>已完成 ${player.completedTurns} 回合</small></span></div>
      <div class="score-box"><strong>${player.score}</strong><small>威望分</small></div>
    </div>
    <div class="assets-row">${columns}<div class="gold-column"><span class="gold-mini">${player.tokens.gold}</span><small>黄金</small></div></div>
    <div class="reserved-section"><div class="reserved-heading"><span>保留的牌</span><span>${player.reserved.length}/3</span></div><div class="reserved-row">${player.reserved.map((entry) => renderReserved(entry, own)).join("") || `<span class="card-id">暂无保留牌</span>`}</div></div>
    <div class="owned-nobles">${nobles}</div>
  </section>`;
}

function renderPlayers() {
  byId("opponent-panel").innerHTML = playerPanel(game.players.opponent, false);
  byId("self-panel").innerHTML = playerPanel(game.players.self, true);
}

function renderHistory() {
  const opponentName = game.opponentLabel.split(" · ")[0];
  byId("history-list").innerHTML = game.history.length
    ? game.history.map((entry) => {
      const actorName = entry.actor === "self" ? "你" : opponentName;
      const details = entry.details.map((detail) => `<li>${html(detail)}</li>`).join("");
      return `<article class="history-entry ${entry.actor}">
        <div class="history-actor"><span>${actorName}</span><small>#${entry.sequence + 1}</small></div>
        <strong>${html(entry.summary)}</strong>
        ${details ? `<ul>${details}</ul>` : ""}
      </article>`;
    }).join("")
    : `<div class="history-empty">暂无行动记录</div>`;
}

function renderActions() {
  byId("phase-label").textContent = PHASE_NAMES[game.phase] || game.phase;
  const titles = {
    main_action: "选择你的主行动",
    payment: "选择支付方案",
    return_tokens: `请归还 ${game.pending?.tokensToReturn || ""} 枚宝石`,
    choose_noble: "选择一位贵族",
    terminal: "对局已结束",
  };
  byId("action-title").textContent = titles[game.phase] || "选择行动";
  const visible = game.phase === "main_action" ? [] : game.legalActions;
  byId("action-list").innerHTML = visible.map((action) => {
    const color = action.color || action.colors?.[0] || "gold";
    return `<button class="action-button ${action.kind.startsWith("Take") ? "token-action" : ""}" style="--action-color:var(--${color})" data-action-index="${action.index}" type="button">${html(action.label)}</button>`;
  }).join("") || (game.phase === "main_action" ? `<span class="card-id">点击宝石堆自由组合拿取；购买和保留请操作对应卡牌或牌堆。</span>` : "");
}

function renderResult() {
  if (!game.result) { byId("result-modal").classList.remove("is-open"); return; }
  const titles = {win: "你获胜了", loss: "对手获胜", draw: "本局平局"};
  byId("result-title").textContent = titles[game.result.outcome];
  byId("result-detail").textContent = `最终比分 ${game.result.humanScore} : ${game.result.opponentScore}，结束原因：${game.result.reason === "final_round" ? "最终轮结算" : "双方连续无合法主行动"}。`;
  byId("result-modal").classList.add("is-open");
}

function render() {
  selectedTokens = [];
  byId("board").classList.remove("is-empty");
  byId("turn-status").textContent = game.result ? "对局结束" : `轮到你行动 · ${PHASE_NAMES[game.phase]}`;
  renderMarket(); renderSupply(); renderNobles(); renderPlayers(); renderHistory(); renderActions(); renderResult();
}

async function api(path, body) {
  const response = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.message || "请求失败");
  return payload;
}

async function loadOpponents() {
  const select = byId("opponent-select");
  const start = byId("start-button");
  start.disabled = true;
  try {
    const response = await fetch("/api/opponents", {cache: "no-store"});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.message || "无法读取对手列表");
    select.innerHTML = payload.opponents.map((opponent) =>
      `<option value="${html(opponent.id)}">${html(opponent.label)}</option>`
    ).join("");
    start.disabled = payload.opponents.length === 0;
  } catch (error) {
    select.innerHTML = `<option value="">读取失败</option>`;
    byId("setup-error").textContent = error.message;
  }
}

function setBusy(value) {
  busy = value;
  if (value) {
    document.querySelectorAll("button:not(:disabled)").forEach((button) => {
      button.disabled = true;
      button.dataset.busyDisabled = "true";
    });
  } else {
    document.querySelectorAll("button[data-busy-disabled]").forEach((button) => {
      button.disabled = false;
      delete button.dataset.busyDisabled;
    });
  }
  if (value && game) byId("turn-status").textContent = "对手思考中…";
}

async function performAction(index) {
  if (busy || !game) return;
  setBusy(true);
  try {
    game = await api(`/api/games/${game.gameId}/actions`, {revision: game.revision, actionIndex: Number(index)});
    render();
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
  }
}

function showToast(message) {
  const toast = byId("toast");
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.setTimeout(() => toast.classList.remove("is-visible"), 3000);
}

document.addEventListener("click", (event) => {
  const token = event.target.closest("[data-token-color]");
  if (token) { selectToken(token.dataset.tokenColor); return; }
  if (event.target.closest("[data-token-reset]")) {
    selectedTokens = [];
    renderSupply();
    return;
  }
  if (event.target.closest("[data-token-confirm]")) {
    const action = exactTokenAction();
    if (action) performAction(action.index);
    return;
  }
  const action = event.target.closest("[data-action-index]");
  if (action) performAction(action.dataset.actionIndex);
});

byId("setup-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const seed = Number(byId("seed-input").value);
  if (!Number.isSafeInteger(seed)) { byId("setup-error").textContent = "Seed 必须是安全整数"; return; }
  setBusy(true);
  byId("setup-error").textContent = "";
  try {
    game = await api("/api/games", {
      opponent: byId("opponent-select").value,
      humanFirst: document.querySelector('input[name="order"]:checked').value === "first",
      seed,
    });
    byId("setup-modal").classList.remove("is-open");
    render();
  } catch (error) {
    byId("setup-error").textContent = error.message;
  } finally {
    setBusy(false);
  }
});

function openSetup() {
  byId("result-modal").classList.remove("is-open");
  byId("setup-modal").classList.add("is-open");
  loadOpponents();
}
byId("new-game-button").addEventListener("click", openSetup);
byId("result-new-game").addEventListener("click", openSetup);
loadOpponents();
