const $ = (id) => document.getElementById(id);
const symbols = {red:'♥', blue:'◆', green:'♣', yellow:'☀', purple:'✦', brown:'▲'};
const storageKey = 'schotten-totten.native-game.v1';
let grant = null, live = null, mode = 'play', selectedCard = null, selectedStone = null;
let busy = false, creating = false, pendingTurn = null, pollTimer = null, pollFlight = null;
let replay = null, replayId = null, frame = 0, playback = null, libraryVersion = 0, replayVersion = 0;
let networkError = '';
let drag = null, suppressCardClickUntil = 0, renderQueued = false;
let recentCards = {opponentPlay:null, drawn:null};

function opponentPolicy(name) {
  const policies = {
    random:{name:'Random', description:'Chooses randomly among legal card placements.'},
    greedy:{name:'Greedy', description:'Scores the available placements and chooses among its best-scoring moves.'},
    builder:{name:'Builder', description:'Evaluates how each placement develops its own formation.'},
    tactician:{name:'Tactician', description:'Evaluates formations across the whole boundary.'},
    planner:{name:'Planner', description:'Considers possible opponent replies to candidate placements.'},
    mcg:{name:'MCG', samples:100, secondsPerSample:0.00075, description:'Monte Carlo search with Greedy rollout play.'},
    mcb:{name:'MCB', samples:20, secondsPerSample:0.005, description:'Monte Carlo search with Builder rollout play.'},
    mcp:{name:'MCP', samples:5, secondsPerSample:0.18, description:'Monte Carlo search with Planner rollout play.'},
  };
  if (!Object.hasOwn(policies, name)) throw new Error('Choose an available opponent.');
  return policies[name];
}
function liveOpponentName(view = live) { return view?.opponent?.name || 'MCB'; }
function updateOpponentSettings() {
  const policy = opponentPolicy($('opponent-choice').value);
  const monteCarlo = policy.samples !== undefined;
  $('samples-group').hidden = !monteCarlo;
  $('samples').disabled = !monteCarlo;
  if (monteCarlo) $('samples').value = policy.samples;
  $('opponent-description').textContent = policy.description;
  updateSamplingWarning();
}
function sampleCount(rawSamples) {
  const value = String(rawSamples).trim();
  if (!/^\d+$/.test(value)) throw new Error('Samples must be a nonnegative whole number.');
  const samples = BigInt(value);
  if (samples > (1n << 64n) - 1n) throw new Error('Samples must fit an unsigned 64-bit integer.');
  return samples.toString();
}
function updateSamplingWarning() {
  const policy = opponentPolicy($('opponent-choice').value);
  const warning = $('sampling-warning');
  warning.hidden = true;
  warning.textContent = '';
  if (policy.samples === undefined) return;
  let samples;
  try { samples = sampleCount($('samples').value); } catch { return; }
  const seconds = Number(samples) * policy.secondsPerSample;
  if (seconds <= 2) return;
  warning.textContent = `This setting is estimated to exceed 2 seconds per turn (about ${seconds.toLocaleString(undefined, {maximumFractionDigits:1})} seconds in the opening on an M4 Pro). Later turns are usually faster; server load can increase the wait. You can still start this game.`;
  warning.hidden = false;
}
function gameRequestBody(opponent, rawSamples, seed, botSeed) {
  const policy = opponentPolicy(opponent);
  const samples = policy.samples === undefined ? null : sampleCount(rawSamples);
  return `{"opponent":${JSON.stringify(opponent)}${policy.samples === undefined ? '' : `,"samples":${samples}`}${seed === null ? '' : `,"seed":${seed}`}${botSeed === null ? '' : `,"bot_seed":${botSeed}`}}`;
}

function node(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}
function notice(message = '') { $('notice').textContent = message; $('notice').hidden = !message; }
function saveGrant() {
  try { localStorage.setItem(storageKey, JSON.stringify(grant)); }
  catch { notice('This browser cannot save the seat token. Keep this page open to continue this game.'); }
}
async function request(path, options = {}, seat = null) {
  const headers = {'Accept':'application/json', ...options.headers};
  if (seat) headers.Authorization = `Bearer ${seat.token}`;
  if (options.body) headers['Content-Type'] = 'application/json';
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(path, {...options, headers, signal:controller.signal, cache:'no-store'});
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      const error = new Error(data?.detail || (typeof data?.error === 'string' ? data.error : data?.error?.message) || data?.message || `The server returned ${response.status}.`);
      error.status = response.status;
      throw error;
    }
    if (!data) throw new Error('The server returned an empty response.');
    return data;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('The server took too long to respond. Reconnect to check your game.');
    throw error;
  } finally { clearTimeout(timer); }
}
function cardKey(c) { return `${c.color}:${c.value}`; }
function cardElement(card, interactive = false, marker = null) {
  const el = node(interactive ? 'button' : 'div', `card ${Object.hasOwn(symbols, card.color) ? card.color : ''}`);
  el.setAttribute('aria-label', `${card.color} ${card.value}`);
  el.append(node('span','value',card.value), node('span','suit',symbols[card.color] || '•'), node('span','center-suit',symbols[card.color] || '•'), node('span','color-label',card.color));
  if (marker) {
    el.classList.add(`recent-${marker}`);
    el.append(node('span', 'card-marker', marker === 'drawn' ? 'NEW' : 'PLAYED'));
    el.setAttribute('aria-label', `${card.color} ${card.value}, ${marker === 'drawn' ? 'newly drawn' : 'opponent’s latest play'}`);
  }
  if (interactive) {
    el.type = 'button';
    el.disabled = !canAct();
    const selected = selectedCard && cardKey(selectedCard) === cardKey(card);
    el.classList.toggle('selected', Boolean(selected));
    el.setAttribute('aria-pressed', String(Boolean(selected)));
    el.draggable = false;
    el.addEventListener('dragstart', event => event.preventDefault());
    el.addEventListener('pointerdown', event => beginCardDrag(event, card));
    el.addEventListener('click', event => {
      if (Date.now() < suppressCardClickUntil || !canAct()) { event.preventDefault(); return; }
      const index = live.hand.findIndex(c => cardKey(c) === cardKey(card));
      selectedCard = selected ? null : card; selectedStone = null; render();
      $('hand').children[index]?.focus();
    });
  }
  return el;
}
function currentView() { return mode === 'play' ? live : replay?.frames[frame]; }
function canAct() { return mode === 'play' && live?.phase === 'your_turn' && !busy && !creating && !pendingTurn && !live.bot_thinking; }
function legalStone(stone) {
  if (stone.claimed_by || stone.yours.length >= 3) return false;
  if (selectedCard && live?.legal_moves) {
    return live.legal_moves.plays.some(play => cardKey(play.card) === cardKey(selectedCard) && play.stones.includes(stone.index));
  }
  return true;
}
function dragIsCurrent(gesture) {
  return canAct() && grant === gesture.grant && live.version === gesture.version && live.hand.some(c => cardKey(c) === cardKey(gesture.card));
}
function dragTarget(gesture, x, y) {
  const lane = document.elementFromPoint(x, y)?.closest('.lane');
  if (!lane || !dragIsCurrent(gesture)) return null;
  const index = Number(lane.dataset.stone), stone = live.stones[index];
  if (!stone || stone.claimed_by || stone.yours.length >= 3) return null;
  const legal = live.legal_moves?.plays.some(play => cardKey(play.card) === cardKey(gesture.card) && play.stones.includes(index));
  return legal ? index : null;
}
function paintCardDrag(gesture, x, y) {
  gesture.x = x; gesture.y = y;
  if (gesture.ghost) gesture.ghost.style.transform = `translate(${x + 14}px, ${y - 55}px)`;
  const target = dragTarget(gesture, x, y);
  for (const lane of document.querySelectorAll('.lane')) {
    const index = Number(lane.dataset.stone);
    const stone = live.stones[index];
    const legal = stone && !stone.claimed_by && stone.yours.length < 3 && live.legal_moves?.plays.some(play => cardKey(play.card) === cardKey(gesture.card) && play.stones.includes(index));
    lane.classList.toggle('drop-legal', Boolean(legal));
    lane.classList.toggle('drop-target', target === index);
  }
}
function activateCardDrag(gesture) {
  if (drag !== gesture || !dragIsCurrent(gesture)) return cancelCardDrag();
  gesture.active = true;
  gesture.source.classList.add('drag-source');
  gesture.ghost = gesture.source.cloneNode(true);
  gesture.ghost.classList.add('drag-ghost');
  gesture.ghost.setAttribute('aria-hidden', 'true');
  gesture.ghost.setAttribute('tabindex', '-1');
  document.body.append(gesture.ghost);
  paintCardDrag(gesture, gesture.x, gesture.y);
  $('drag-status').textContent = 'Drop on a highlighted stone to play immediately.';
}
function beginCardDrag(event, card) {
  if (!canAct() || drag || event.isPrimary === false || event.button !== 0) return;
  const gesture = {pointerId:event.pointerId, pointerType:event.pointerType, source:event.currentTarget, card, grant, version:live.version, startX:event.clientX, startY:event.clientY, x:event.clientX, y:event.clientY, active:false};
  drag = gesture;
  gesture.source.setPointerCapture(event.pointerId);
  if (event.pointerType === 'touch') gesture.timer = setTimeout(() => activateCardDrag(gesture), 220);
}
function moveCardDrag(event) {
  const gesture = drag;
  if (!gesture || event.pointerId !== gesture.pointerId) return;
  if (!dragIsCurrent(gesture)) return cancelCardDrag();
  const distance = Math.hypot(event.clientX - gesture.startX, event.clientY - gesture.startY);
  gesture.x = event.clientX; gesture.y = event.clientY;
  if (!gesture.active && distance >= 7) {
    if (gesture.pointerType === 'touch') return cancelCardDrag();
    activateCardDrag(gesture);
  }
  if (gesture.active) { event.preventDefault(); paintCardDrag(gesture, event.clientX, event.clientY); }
}
function cancelCardDrag(flush = true) {
  const gesture = drag;
  if (!gesture) return;
  drag = null;
  clearTimeout(gesture.timer);
  if (gesture.active) suppressCardClickUntil = Date.now() + 700;
  gesture.ghost?.remove();
  gesture.source.classList.remove('drag-source');
  if (gesture.source.hasPointerCapture(gesture.pointerId)) gesture.source.releasePointerCapture(gesture.pointerId);
  for (const lane of document.querySelectorAll('.lane')) { lane.classList.remove('drop-legal'); lane.classList.remove('drop-target'); }
  $('drag-status').textContent = '';
  const shouldRender = renderQueued; renderQueued = false;
  if (flush && shouldRender) render();
}
function endCardDrag(event) {
  const gesture = drag;
  if (!gesture || event.pointerId !== gesture.pointerId) return;
  const target = gesture.active ? dragTarget(gesture, event.clientX, event.clientY) : null;
  const active = gesture.active;
  cancelCardDrag(false);
  if (active && target !== null) { selectedCard = gesture.card; selectedStone = target; submitTurn(); }
  else if (active) render();
}
function renderRetryControl(watching) {
  const saveError = live?.phase === 'game_over' && live.recording_error;
  $('retry').hidden = watching || (!networkError && !pendingTurn && !saveError);
  $('retry').disabled = busy || creating;
  $('retry').textContent = pendingTurn ? 'Check / retry move' : saveError ? 'Retry saving replay' : 'Reconnect';
}
async function retryLive() {
  await refreshLive();
  if (pendingTurn && !busy) await submitTurn();
}
function seatName(index) { const seat = replay?.metadata?.seats?.[index]; return typeof seat === 'string' ? seat : seat?.name || `Seat ${index + 1}`; }
function reasonText(reason) { return String(reason || 'Game complete').replaceAll('_',' '); }
function render() {
  if (drag) { renderQueued = true; return; }
  const view = currentView();
  const watching = mode === 'replay';
  $('play-mode').classList.toggle('active', !watching); $('replay-mode').classList.toggle('active', watching);
  $('play-mode').setAttribute('aria-pressed', String(!watching)); $('replay-mode').setAttribute('aria-pressed', String(watching));
  $('mode-label').textContent = watching ? 'EVERY MOVE HAS A STORY' : 'A GAME OF SMALL BATTLES';
  $('headline').textContent = watching ? 'A seat at the archive.' : 'Nine stones. One boundary.';
  $('library').hidden = !watching;
  $('play-controls').hidden = watching;
  $('replay-controls').hidden = !watching || !replay;
  $('opponent-name').textContent = watching ? seatName(1) : liveOpponentName();
  $('hand-title').textContent = watching ? `${seatName(0)} · hand` : 'Your hand';
  $('their-score-label').textContent = watching ? 'Seat 2 stones' : `${liveOpponentName()} stones`;
  $('your-score-label').textContent = watching ? 'Seat 1 stones' : 'Your stones';
  $('opponent-status').textContent = watching ? 'Revealed hand · recorded game' : live?.bot_thinking ? `${liveOpponentName()} is thinking…` : live?.bot_error ? `${liveOpponentName()} paused · see message above` : live?.phase === 'your_turn' ? `${liveOpponentName()} is ready for your move` : `Native ${liveOpponentName()} opponent`;
  $('their-score').textContent = view?.stones.filter(s => s.claimed_by === 'them').length || 0;
  $('your-score').textContent = view?.stones.filter(s => s.claimed_by === 'you').length || 0;
  $('deck').textContent = view?.deck_count ?? '—';
  $('hand-count').textContent = view ? `${view.hand.length} cards` : '';
  const updates = [];
  if (!watching && recentCards.opponentPlay) {
    const {card, stone} = recentCards.opponentPlay;
    updates.push(`${liveOpponentName()} played ${card.color} ${card.value} at stone ${stone + 1}.`);
  }
  if (!watching && recentCards.drawn) updates.push(`You drew ${recentCards.drawn.color} ${recentCards.drawn.value}.`);
  $('recent-turn').hidden = updates.length === 0;
  $('recent-turn').textContent = updates.join(' ');
  $('board').replaceChildren();
  for (let i = 0; i < 9; i++) {
    const stone = view?.stones[i] || {yours:[], theirs:[], claimed_by:null};
    const lane = node('div','lane');
    lane.dataset.stone = i;
    for (const side of ['theirs', 'stone', 'yours']) {
      if (side === 'stone') {
        const button = node('button',`stone ${stone.claimed_by || ''}`,String(i + 1));
        button.type = 'button';
        const legal = canAct() && Boolean(selectedCard) && legalStone(stone);
        button.disabled = !legal;
        button.classList.toggle('legal', legal);
        button.classList.toggle('selected', selectedStone === i && !watching);
        button.setAttribute('aria-pressed', String(selectedStone === i && !watching));
        const owner = stone.claimed_by === 'you' ? (watching ? seatName(0) : 'you') : stone.claimed_by === 'them' ? (watching ? seatName(1) : liveOpponentName()) : 'unclaimed';
        button.setAttribute('aria-label', `Stone ${i + 1}, ${owner}, ${stone.yours.length} cards below, ${stone.theirs.length} cards above${legal ? ', available' : ''}`);
        if (stone.claimed_by) button.append(node('span','owner',stone.claimed_by === 'you' ? '▼' : '▲'));
        button.addEventListener('click', () => { selectedStone = i; render(); $('confirm').focus(); });
        lane.append(button);
      } else {
        const formation = node('div',`formation ${side}`);
        if (!stone[side].length) formation.append(node('div','slot'));
        for (const card of stone[side]) {
          const latest = recentCards.opponentPlay;
          const marker = !watching && side === 'theirs' && latest?.stone === i && cardKey(latest.card) === cardKey(card) ? 'played' : null;
          formation.append(cardElement(card, false, marker));
        }
        lane.append(formation);
      }
    }
    $('board').append(lane);
  }
  $('hand').replaceChildren(...(view?.hand || []).map(card => cardElement(card, !watching, !watching && recentCards.drawn && cardKey(card) === cardKey(recentCards.drawn) ? 'drawn' : null)));
  $('opponent-hand').replaceChildren();
  if (watching && view) {
    for (const card of view.opponent_hand || []) $('opponent-hand').append(cardElement(card));
  } else if (view) {
    for (let i = 0; i < view.opponent_hand_count; i++) $('opponent-hand').append(node('div','card-back'));
    $('opponent-hand').setAttribute('aria-label', `${liveOpponentName()} has ${view.opponent_hand_count} hidden cards`);
  }
  if (watching) $('opponent-hand').setAttribute('aria-label', `${seatName(1)} revealed hand`);
  const done = view?.phase === 'game_over';
  $('result').hidden = !done;
  if (done) {
    const winner = view.winner === 'you' ? (watching ? seatName(0) : 'You') : view.winner === 'them' ? (watching ? seatName(1) : liveOpponentName()) : null;
    $('result').textContent = `${winner ? `${winner} ${winner === 'You' ? 'win' : 'wins'} the boundary` : 'The game is a draw'} · ${reasonText(view.win_reason)}`;
  }
  $('board-hint').textContent = watching ? `${seatName(0)} below · ${seatName(1)} above` : `Your formations below · ${liveOpponentName()} above`;
  $('confirm').disabled = !canAct() || !selectedCard || selectedStone === null;
  $('recording-status').hidden = watching || !done;
  $('recording-status').textContent = !watching && done ? view.recording_error ? `Replay could not be saved: ${view.recording_error}` : view.recording_saved ? 'Replay saved in Recordings' : 'Saving replay…' : '';
  $('confirm').textContent = busy ? 'Sending move…' : 'Play card';
  $('pass').hidden = !canAct() || (live.legal_moves ? live.legal_moves.plays.length > 0 : live.hand.length > 0 && live.stones.some(legalStone));
  renderRetryControl(watching);
  $('turn-label').textContent = done ? 'BOUNDARY SETTLED' : busy ? 'SENDING YOUR MOVE' : pendingTurn ? 'CHECKING YOUR MOVE' : view?.bot_thinking || view?.phase === 'opponent_turn' ? `${liveOpponentName().toUpperCase()} IS THINKING` : canAct() ? 'YOUR TURN' : grant ? 'RESTORING YOUR SEAT' : 'WELCOME TO THE TABLE';
  $('selection').textContent = done ? 'A new game is always a fresh start.' : pendingTurn ? 'Your move may have arrived. Reconnect to confirm it safely.' : selectedCard ? `${selectedCard.color} ${selectedCard.value}${selectedStone === null ? ' · choose a highlighted stone' : ` → stone ${selectedStone + 1}`}` : canAct() ? 'Choose a card to begin your formation.' : grant ? 'Waiting for the table…' : 'Start a game to take your seat.';
  if (watching && replay) renderReplayControls();
}
function trackRecentCards(previous, view) {
  if (view.recent_cards) {
    recentCards = view.recent_cards;
    return;
  }
  if (!previous || previous.game_id !== view.game_id) {
    recentCards = {opponentPlay:null, drawn:null};
    return;
  }
  if (view.version <= previous.version) return;
  const oldHand = new Set(previous.hand.map(cardKey));
  const drawn = view.hand.find(card => !oldHand.has(cardKey(card)));
  if (drawn) recentCards.drawn = drawn;
  else if (recentCards.drawn && !view.hand.some(card => cardKey(card) === cardKey(recentCards.drawn))) recentCards.drawn = null;
  view.stones.forEach((stone, index) => {
    const oldCards = new Set((previous.stones[index]?.theirs || []).map(cardKey));
    const played = stone.theirs.find(card => !oldCards.has(cardKey(card)));
    if (played) recentCards.opponentPlay = {card:played, stone:index};
  });
}
function acceptView(view) {
  if (live && view.version < live.version) return;
  const changed = !live || view.version !== live.version || view.bot_thinking !== live.bot_thinking || view.bot_error !== live.bot_error || view.opponent?.name !== live.opponent?.name || view.opponent?.strategy !== live.opponent?.strategy || view.recording_saved !== live.recording_saved || view.recording_error !== live.recording_error || view.recording_id !== live.recording_id || Boolean(networkError) || Boolean(pendingTurn && view.version > pendingTurn.version);
  if (drag && changed) cancelCardDrag(false);
  trackRecentCards(live, view);
  live = view;
  if (pendingTurn && view.version > pendingTurn.version) pendingTurn = null;
  if (selectedCard && !view.hand.some(c => cardKey(c) === cardKey(selectedCard))) { selectedCard = null; selectedStone = null; }
  if (selectedStone !== null && !legalStone(view.stones[selectedStone])) selectedStone = null;
  networkError = '';
  if (mode === 'play' && changed) { notice(view.bot_error ? `${liveOpponentName(view)} could not finish its move: ${view.bot_error}. Start a new game to continue playing.` : ''); render(); }
}
function schedulePoll(delay = 900) {
  clearTimeout(pollTimer);
  if (mode === 'play' && grant && (!live || live.phase !== 'game_over' || (!live.recording_saved && !live.recording_error))) pollTimer = setTimeout(() => refreshLive(), delay);
}
async function refreshLive() {
  if (!grant || pollFlight === grant) return;
  const seat = grant;
  pollFlight = seat;
  try {
    const view = await request(`/web/games/${encodeURIComponent(seat.game_id)}`, {}, seat);
    if (grant !== seat) return;
    acceptView(view);
  } catch (error) {
    if (grant !== seat) return;
    networkError = error.message;
    if (mode === 'play') {
      notice((error.status === 404 || error.status === 401) ? 'This saved seat is no longer available. The local server may have restarted. Start a new game to play again.' : `Cannot reach the table. ${error.message}`);
      render();
    }
  } finally {
    if (pollFlight === seat) pollFlight = null;
    if (grant === seat) schedulePoll(networkError || live?.bot_error ? 4000 : 900);
  }
}
async function submitTurn(pass = false) {
  if ((!canAct() && !pendingTurn) || busy || creating || !grant) return;
  const seat = grant;
  if (!pendingTurn) pendingTurn = {version:live.version, turn_id:uuid(), play:pass ? null : {card:selectedCard, stone:selectedStone}, claims:[]};
  const body = pendingTurn;
  busy = true; render();
  try {
    const view = await request(`/web/games/${encodeURIComponent(seat.game_id)}/turn`, {method:'POST', body:JSON.stringify(body)}, seat);
    if (grant !== seat) return;
    pendingTurn = null; selectedCard = null; selectedStone = null; acceptView(view);
  } catch (error) {
    if (grant !== seat) return;
    if (error.status && error.status < 500) pendingTurn = null;
    networkError = error.message;
    if (mode === 'play') notice(`Move not confirmed. ${error.message}`);
  } finally {
    if (grant === seat) { busy = false; render(); schedulePoll(100); }
  }
}
function uuid() {
  if (globalThis.crypto?.randomUUID) return crypto.randomUUID();
  const bytes = new Uint8Array(16); crypto.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128;
  const hex = Array.from(bytes, b => b.toString(16).padStart(2,'0')).join('');
  return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`;
}
function seedValue(id) {
  const value = $(id).value.trim();
  if (!value) return null;
  if (!/^-?\d+$/.test(value)) throw new Error('Seeds must be signed whole numbers.');
  const number = BigInt(value);
  if (number < -(1n << 63n) || number > (1n << 63n) - 1n) throw new Error('Seeds must fit a signed 64-bit integer.');
  return number.toString();
}
$('opponent-choice').addEventListener('change', updateOpponentSettings);
$('samples').addEventListener('input', updateSamplingWarning);
updateOpponentSettings();
$('settings-form').addEventListener('submit', async event => {
  event.preventDefault(); if (creating) return;
  $('settings-error').textContent = '';
  let body;
  try {
    const seed = seedValue('seed'), botSeed = seedValue('bot-seed');
    body = gameRequestBody($('opponent-choice').value, $('samples').value, seed, botSeed);
  } catch (error) { $('settings-error').textContent = error.message; return; }
  cancelCardDrag();
  creating = true; $('create').disabled = true; $('create').textContent = 'Dealing…'; render();
  try {
    const created = await request('/web/games', {method:'POST', body});
    if (!created.game_id || !created.token) throw new Error('The server did not return a seat token.');
    grant = created; live = null; recentCards = {opponentPlay:null, drawn:null}; pendingTurn = null; selectedCard = null; selectedStone = null; busy = false;
    saveGrant(); stopPlayback(); mode = 'play'; ++replayVersion; notice(); $('settings').close();
    await refreshLive();
  } catch (error) { $('settings-error').textContent = error.message; }
  finally { creating = false; $('create').disabled = false; $('create').textContent = 'Deal a new game'; render(); }
});
function switchMode(next) {
  cancelCardDrag(false);
  mode = next; ++replayVersion; stopPlayback(); clearTimeout(pollTimer); notice(); render();
  if (next === 'play') { if (grant) refreshLive(); }
  else loadLibrary();
}
async function loadLibrary() {
  const version = ++libraryVersion;
  $('library-status').textContent = 'Opening the archive…'; $('refresh').disabled = true;
  try {
    const data = await request('/web/recordings');
    if (version !== libraryVersion) return;
    $('recording-list').replaceChildren();
    $('library-status').textContent = data.games.length ? `${data.games.length} recorded ${data.games.length === 1 ? 'game' : 'games'}${data.warnings?.length ? ` · ${data.warnings.length} unreadable files skipped.\n${data.warnings.join('\n')}` : ''}` : `No recordings yet. Finish a game against any opponent to save a replay automatically, or run a tournament with recording enabled. Then refresh the library.${data.warnings?.length ? `\n${data.warnings.join('\n')}` : ''}`;
    for (const game of data.games) {
      const names = (game.seats || []).map((seat, i) => typeof seat === 'string' ? seat : seat.name || `Seat ${i + 1}`);
      const item = node('button','recording-item'); item.type = 'button';
      item.classList.toggle('selected', game.id === replayId);
      item.setAttribute('aria-pressed', String(game.id === replayId));
      item.append(node('strong','',names.join(' vs ') || game.game_id), node('small','',(game.seats || []).map(s => s.strategy).filter(Boolean).join(' · ')),node('small','',`${game.turns} turns · ${game.winner ?? 'Draw'} · ${reasonText(game.reason)}`),node('small','',game.id));
      item.addEventListener('click', () => loadReplay(game.id, item));
      $('recording-list').append(item);
    }
  } catch (error) { if (version === libraryVersion) $('library-status').textContent = `The archive is unavailable. ${error.message}`; }
  finally { if (version === libraryVersion) $('refresh').disabled = false; }
}
async function loadReplay(id, item) {
  const version = ++replayVersion; stopPlayback(); notice('Loading and validating the recording…');
  try {
    const path = id.split('/').map(encodeURIComponent).join('/');
    const data = await request(`/web/recordings/${path}`);
    if (version !== replayVersion || mode !== 'replay') return;
    if (!Array.isArray(data.frames) || !data.frames.length) throw new Error('This recording has no positions.');
    replay = data; replayId = id; frame = 0; notice(); render();
    for (const entry of $('recording-list').children) { entry.classList.toggle('selected', entry === item); entry.setAttribute('aria-pressed', String(entry === item)); }
  } catch (error) { if (version === replayVersion && mode === 'replay') notice(`Cannot open this recording. ${error.message}`); }
}
function renderReplayControls() {
  const metadata = replay.metadata;
  const strategies = (metadata.seats || []).map(s => s.strategy).filter(Boolean).join(' vs ');
  $('replay-meta').textContent = `${seatName(0)} vs ${seatName(1)}${strategies ? ` · ${strategies}` : ''} · ${metadata.winner ?? 'Draw'} · ${reasonText(metadata.reason)} · ${metadata.turns} turns`;
  $('frame-label').textContent = `Turn ${frame} / ${replay.frames.length - 1}`;
  $('seek').max = replay.frames.length - 1; $('seek').value = frame;
  $('first').disabled = $('previous').disabled = frame === 0;
  $('next').disabled = $('last').disabled = frame === replay.frames.length - 1;
  $('autoplay').disabled = replay.frames.length < 2;
  $('autoplay').textContent = playback ? 'Pause replay' : 'Play replay';
  const move = replay.frames[frame].last_turn;
  $('last-move').textContent = !move ? 'The initial deal. Both hands are revealed in replay.' : `${seatName(move.seat)} ${move.play ? `played ${move.play.card.color} ${move.play.card.value} at stone ${move.play.stone + 1}` : 'passed'}${move.claims?.length ? ` · claimed ${move.claims.map(i => i + 1).join(', ')}` : ''}.`;
}
function stopPlayback() { clearTimeout(playback); playback = null; if (replay) $('autoplay').textContent = 'Play replay'; }
function seek(next) { stopPlayback(); if (!replay) return; frame = Math.max(0, Math.min(replay.frames.length - 1, Number(next))); render(); }
function tick() {
  if (mode !== 'replay' || !replay) return stopPlayback();
  playback = setTimeout(() => {
    if (frame >= replay.frames.length - 1) { stopPlayback(); render(); return; }
    frame++;
    if (frame === replay.frames.length - 1) stopPlayback(); else tick();
    render();
  }, Number($('speed').value));
}
$('play-mode').addEventListener('click', () => switchMode('play'));
$('replay-mode').addEventListener('click', () => switchMode('replay'));
$('new-game').addEventListener('click', () => { cancelCardDrag(); $('settings-error').textContent = ''; $('settings').showModal(); });
$('close-settings').addEventListener('click', () => $('settings').close());
$('confirm').addEventListener('click', () => { if (selectedCard && selectedStone !== null) submitTurn(); });
$('pass').addEventListener('click', () => submitTurn(true));
$('retry').addEventListener('click', retryLive);
$('refresh').addEventListener('click', loadLibrary);
$('first').addEventListener('click', () => seek(0));
$('previous').addEventListener('click', () => seek(frame - 1));
$('next').addEventListener('click', () => seek(frame + 1));
$('last').addEventListener('click', () => seek(replay.frames.length - 1));
$('seek').addEventListener('input', event => seek(event.target.value));
$('autoplay').addEventListener('click', () => { if (playback) stopPlayback(); else { if (frame === replay.frames.length - 1) frame = 0; tick(); } render(); });
$('speed').addEventListener('change', () => { if (playback) { stopPlayback(); tick(); } });
document.addEventListener('pointermove', moveCardDrag, {passive:false});
document.addEventListener('pointerup', endCardDrag);
document.addEventListener('pointercancel', () => cancelCardDrag());
document.addEventListener('lostpointercapture', () => { if (drag) cancelCardDrag(); });
document.addEventListener('touchmove', event => { if (drag?.active) event.preventDefault(); }, {passive:false});
document.addEventListener('keydown', event => { if (event.key === 'Escape') cancelCardDrag(); });
window.addEventListener('blur', () => cancelCardDrag());
window.addEventListener('pagehide', () => { cancelCardDrag(); clearTimeout(pollTimer); stopPlayback(); });
window.addEventListener('pageshow', () => { if (mode === 'play' && grant) refreshLive(); });
try { const saved = JSON.parse(localStorage.getItem(storageKey)); if (saved && typeof saved.game_id === 'string' && typeof saved.token === 'string') grant = saved; } catch { /* A blocked or stale store is safe to ignore. */ }
render();
if (grant) refreshLive(); else $('settings').showModal();
