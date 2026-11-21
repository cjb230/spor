const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');

const app = readFileSync(join(__dirname, 'app.js'), 'utf8');
// Run the actual view acceptance function with a small rendering boundary.
// No browser DOM is needed: a render replaces focus, so its invocation is observable.
const acceptView = app.slice(app.indexOf('function acceptView('), app.indexOf('function schedulePoll('));
function harness() {
  const context = vm.createContext({});
  vm.runInContext(`
    let live = null, pendingTurn = null, selectedCard = null, selectedStone = null;
    let networkError = '', mode = 'play', renders = 0;
    let drag = null; function cancelCardDrag() {}
    let recentCards = {opponentPlay:null, drawn:null};
    function cardKey(card) { return card.color+':'+card.value; }
    ${app.slice(app.indexOf('function trackRecentCards('), app.indexOf('function acceptView('))}
    let focus = 'selected-hand-card';
    function render() { renders++; focus = null; }
    function notice() {}
    function liveOpponentName(){return "MCB"}
    ${acceptView}
    const view = {version:1, hand:[], stones:[], bot_thinking:false, bot_error:null};
  `, context);
  return (code) => vm.runInContext(code, context);
}

test('unchanged polling preserves the DOM and keyboard focus', () => {
  const run = harness();
  run('acceptView(view); focus = "selected-hand-card";');
  run('acceptView({...view}); acceptView({...view});');
  assert.equal(run('renders'), 1);
  assert.equal(run('focus'), 'selected-hand-card');
});

test('thinking and bot error changes update the rendered state', () => {
  const run = harness();
  run('acceptView(view); acceptView({...view, bot_thinking:true});');
  assert.equal(run('renders'), 2);
  run('acceptView({...view, bot_error:"search failed"});');
  assert.equal(run('renders'), 3);
});

test('network recovery rerenders even when the game version is unchanged', () => {
  const run = harness();
  run('acceptView(view); networkError = "offline"; acceptView({...view});');
  assert.equal(run('renders'), 2);
  assert.equal(run('networkError'), '');
});

test('an accepted pending move clears its retry and renders the next position', () => {
  const run = harness();
  run('acceptView(view); pendingTurn = {version:1}; acceptView({...view,version:2});');
  assert.equal(run('pendingTurn'), null);
  assert.equal(run('live.version'), 2);
  assert.equal(run('renders'), 2);
});

test('late responses cannot replace a newer position or move keyboard focus', () => {
  const run = harness();
  run('acceptView({...view,version:2}); focus = "selected-hand-card"; acceptView(view);');
  assert.equal(run('live.version'), 2);
  assert.equal(run('renders'), 1);
  assert.equal(run('focus'), 'selected-hand-card');
});

test('opponent plays and draws are detected even when polling skips the intermediate turn', () => {
  const run = harness();
  run(`
    const red={color:'red',value:4},blue={color:'blue',value:7},green={color:'green',value:2};
    const initial={...view,version:1,hand:[red],stones:[{theirs:[],yours:[]}]};
    acceptView(initial);
  `);
  assert.equal(run('recentCards.drawn'),null);
  run("acceptView({...initial,version:3,hand:[green],stones:[{theirs:[blue],yours:[red]}]})");
  assert.equal(run('cardKey(recentCards.drawn)'),'green:2');
  assert.equal(run('cardKey(recentCards.opponentPlay.card)'),'blue:7');
  assert.equal(run('recentCards.opponentPlay.stone'),0);
  run('acceptView({...live}); acceptView({...initial,version:2});');
  assert.equal(run('cardKey(recentCards.drawn)'),'green:2');
  assert.equal(run('cardKey(recentCards.opponentPlay.card)'),'blue:7');
});

test('draw markers persist through the bot reply and clear when played with an empty deck', () => {
  const run = harness();
  run(`
    const red={color:'red',value:4},blue={color:'blue',value:7};
    acceptView({...view,hand:[red],stones:[{theirs:[],yours:[]}]});
    acceptView({...live,version:2,hand:[blue],stones:[{theirs:[],yours:[red]}]});
    acceptView({...live,version:3,stones:[{theirs:[red],yours:[red]}]});
  `);
  assert.equal(run('cardKey(recentCards.drawn)'),'blue:7');
  run('acceptView({...live,version:4,hand:[]})');
  assert.equal(run('recentCards.drawn'),null);
  run('live=null;acceptView({...view,version:1})');
  assert.equal(run('recentCards.opponentPlay'),null);
  assert.equal(run('recentCards.drawn'),null);
});

test('a restored page recovers recent cards from the server without observing the turn', () => {
  const run = harness();
  run(`acceptView({...view,recent_cards:{opponentPlay:{card:{color:'red',value:2},stone:8},drawn:{color:'blue',value:5}}})`);
  assert.equal(run('cardKey(recentCards.drawn)'), 'blue:5');
  assert.equal(run('recentCards.opponentPlay.stone'), 8);
  run('acceptView({...live,recording_saved:true})');
  assert.equal(run('cardKey(recentCards.opponentPlay.card)'), 'red:2');
});

test('recent cards have visible badges and descriptive accessible names', () => {
  const context = vm.createContext({});
  vm.runInContext(`
    const symbols={blue:'◆'};
    const document={createElement(){
      const element={children:[],attributes:{},classes:[],
        setAttribute(name,value){this.attributes[name]=value},append(...children){this.children.push(...children)}};
      element.classList={add(value){element.classes.push(value)}};
      return element;
    }};
    ${app.slice(app.indexOf('function node('), app.indexOf('function notice('))}
    ${app.slice(app.indexOf('function cardElement('), app.indexOf('function currentView('))}
  `, context);
  for (const [marker,badge,description] of [['played','PLAYED','opponent’s latest play'],['drawn','NEW','newly drawn']]) {
    vm.runInContext(`
      var card=cardElement({color:'blue',value:7},false,'${marker}');
    `,context);
    assert.equal(vm.runInContext(`card.classes.includes('recent-${marker}')`,context),true);
    assert.equal(vm.runInContext('card.children.at(-1).textContent',context),badge);
    assert.equal(vm.runInContext("card.attributes['aria-label']",context),`blue 7, ${description}`);
  }
  vm.runInContext("var plain=cardElement({color:'blue',value:7})",context);
  assert.equal(vm.runInContext('plain.children.length',context),4);
});

function dragHarness() {
  const context = vm.createContext({setTimeout, clearTimeout});
  const handlers = app.slice(app.indexOf('function dragIsCurrent('), app.indexOf('function seatName('));
  vm.runInContext(`
    let grant={game_id:'a'},live={version:1,phase:'your_turn',hand:[{color:'red',value:3}],stones:[{index:0,yours:[],claimed_by:null}],legal_moves:{plays:[{card:{color:'red',value:3},stones:[0]}]}};
    let mode='play',busy=false,creating=false,pendingTurn=null,selectedCard=null,selectedStone=null;
    let drag=null,suppressCardClickUntil=0,renderQueued=false,submits=0,renders=0;
    const classes={add(){},remove(){},toggle(){}};
    const source={classList:classes,setPointerCapture(){},hasPointerCapture(){return true},releasePointerCapture(){},cloneNode(){return {classList:classes,style:{},setAttribute(){},remove(){}}}};
    const lane={dataset:{stone:'0'},classList:classes};
    let hit=lane;
    const document={body:{append(){}},querySelectorAll(){return [lane]},elementFromPoint(){return hit?{closest(){return hit}}:null}};
    function $(id){return {textContent:''}}
    function cardKey(c){return c.color+':'+c.value}
    function canAct(){return mode==='play'&&live.phase==='your_turn'&&!busy&&!creating&&!pendingTurn&&!live.bot_thinking}
    function render(){renders++}
    function submitTurn(){submits++;busy=true}
    ${handlers}
    function event(x=0,y=0){return {pointerId:1,pointerType:'mouse',button:0,isPrimary:true,clientX:x,clientY:y,currentTarget:source,preventDefault(){}}}
    const card=live.hand[0];
  `,context);
  return code=>vm.runInContext(code,context);
}

test('dragging to a legal lane submits exactly once without rendering the source away',()=>{
  const run=dragHarness();
  run('beginCardDrag(event(),card); moveCardDrag(event(20,20));');
  assert.equal(run('renders'),0);
  run('endCardDrag(event(20,20)); endCardDrag(event(20,20));');
  assert.equal(run('submits'),1);
  assert.equal(run('selectedStone'),0);
  assert.equal(run('suppressCardClickUntil > Date.now()'),true);
});

test('outside, illegal and canceled card drags never submit',()=>{
  for(const finish of ['hit=null; endCardDrag(event(20,20))','live.stones[0].claimed_by="them"; endCardDrag(event(20,20))','cancelCardDrag()']){
    const run=dragHarness();run('beginCardDrag(event(),card); moveCardDrag(event(20,20));');run(finish);
    assert.equal(run('submits'),0);
  }
});

test('stale game, version, mode and thinking state invalidate a drag',()=>{
  for(const mutate of ['grant={game_id:"b"}','live.version++','mode="replay"','live.bot_thinking=true']){
    const run=dragHarness();run('beginCardDrag(event(),card); moveCardDrag(event(20,20));');run(mutate);run('endCardDrag(event(20,20));');
    assert.equal(run('submits'),0);
  }
});

test('ordinary clicks remain clicks and input is blocked while busy',()=>{
  const run=dragHarness();run('beginCardDrag(event(),card); endCardDrag(event());');
  assert.equal(run('submits'),0);
  assert.equal(run('suppressCardClickUntil'),0);
  run('busy=true; beginCardDrag(event(),card);');
  assert.equal(run('drag'),null);
});

test('touch movement before the hold threshold keeps ordinary scrolling available',()=>{
  const run=dragHarness();
  run('const touch=event();touch.pointerType="touch";beginCardDrag(touch,card);moveCardDrag(event(0,20));');
  assert.equal(run('drag'),null);
  assert.equal(run('submits'),0);
  assert.equal(run('suppressCardClickUntil'),0);
});

test('a held touch can drag and drop a card',async()=>{
  const run=dragHarness();
  run('const touch=event();touch.pointerType="touch";beginCardDrag(touch,card);');
  await new Promise(resolve=>setTimeout(resolve,240));
  assert.equal(run('drag.active'),true);
  run('moveCardDrag(event(20,20));endCardDrag(event(20,20));');
  assert.equal(run('submits'),1);
});

test('recording save completion and errors rerender the final-game status',()=>{
  const run=harness();
  run('acceptView({...view,recording_saved:false,recording_error:null});acceptView({...view,recording_saved:true,recording_error:null,recording_id:"run/game.json"});');
  assert.equal(run('renders'),2);
  run('acceptView({...view,recording_saved:false,recording_error:"disk full"});');
  assert.equal(run('renders'),3);
});

test('final replay save errors expose a retry button that refreshes the live game',async()=>{
  const context=vm.createContext({});
  const helper=app.slice(app.indexOf('function renderRetryControl('),app.indexOf('function seatName('));
  vm.runInContext(`
    let live={phase:'game_over',recording_error:'disk full'},networkError='',pendingTurn=null,busy=false,creating=false,refreshes=0;
    const button={};function $(){return button}
    async function refreshLive(){refreshes++}
    async function submitTurn(){throw new Error('Saving a replay must not submit a turn')}
    ${helper}
  `,context);
  vm.runInContext('renderRetryControl(false)',context);
  assert.equal(vm.runInContext('button.hidden',context),false);
  assert.equal(vm.runInContext('button.textContent',context),'Retry saving replay');
  await vm.runInContext('retryLive()',context);
  assert.equal(vm.runInContext('refreshes',context),1);
  vm.runInContext('renderRetryControl(true)',context);
  assert.equal(vm.runInContext('button.hidden',context),true);
});

function opponentHarness(){
  const context=vm.createContext({});
  const helpers=app.slice(app.indexOf('function opponentPolicy('),app.indexOf('function node('));
  vm.runInContext(`
    let live=null;
    const elements={'opponent-choice':{value:'mcb'},'samples':{value:'20'},'samples-group':{},'opponent-description':{},'sampling-warning':{}};
    function $(id){return elements[id]}
    ${helpers}
  `,context);
  return code=>vm.runInContext(code,context);
}

test('each Monte Carlo opponent has its own sample default and sends the selected policy',()=>{
  const run=opponentHarness();
  for(const [opponent,samples] of [['mcg',100],['mcb',20],['mcp',5]]){
    run(`elements['opponent-choice'].value='${opponent}';updateOpponentSettings();`);
    assert.equal(Number(run("elements.samples.value")),samples);
    assert.equal(run("elements['samples-group'].hidden"),false);
    assert.equal(run('elements.samples.disabled'),false);
    const body=JSON.parse(run(`gameRequestBody('${opponent}',${samples},null,null)`));
    assert.deepEqual(body,{opponent,samples});
  }
});

test('fixed opponents hide and disable irrelevant samples and omit them from creation',()=>{
  const run=opponentHarness();
  for(const opponent of ['random','greedy','builder','tactician','planner']){
    run(`elements['opponent-choice'].value='${opponent}';updateOpponentSettings();`);
    assert.equal(run("elements['samples-group'].hidden"),true);
    assert.equal(run('elements.samples.disabled'),true);
    assert.deepEqual(JSON.parse(run(`gameRequestBody('${opponent}','invalid',null,null)`)),{opponent});
    assert.ok(run("elements['opponent-description'].textContent.length")>10);
  }
});

test('live opponent labels use the server view, independent of dialog choice and reload',()=>{
  const run=opponentHarness();
  run("live={opponent:{name:'Greedy',strategy:'greedy'}};elements['opponent-choice'].value='mcp';updateOpponentSettings();");
  assert.equal(run('liveOpponentName()'),'Greedy');
  run("live=JSON.parse(JSON.stringify({opponent:{name:'MCG',strategy:'mcg:100'}}))");
  assert.equal(run('liveOpponentName()'),'MCG');
  run('live={}');assert.equal(run('liveOpponentName()'),'MCB');
});

test('MC samples have no arbitrary cap and serialization preserves wide integers',()=>{
  const run=opponentHarness();
  for (const samples of ['0','101','1000001','18446744073709551615']) {
    assert.match(run(`gameRequestBody('mcb','${samples}',null,null)`),new RegExp(`"samples":${samples}[}]`));
  }
  for (const samples of ['','-1','1.5','Infinity','18446744073709551616']) {
    assert.throws(()=>run(`gameRequestBody('mcb','${samples}',null,null)`));
  }
  assert.throws(()=>run("gameRequestBody('unknown',20,null,null)"));
  assert.match(run("gameRequestBody('random','ignored','9223372036854775807','-9223372036854775808')"),/"seed":9223372036854775807,"bot_seed":-9223372036854775808/);
});

test('sampling warnings update above two seconds without preventing creation',()=>{
  const run=opponentHarness();
  run('updateOpponentSettings()');
  assert.equal(run("elements['sampling-warning'].hidden"),true);
  for (const opponent of ['mcg','mcb','mcp']) {
    run(`elements['opponent-choice'].value='${opponent}';updateOpponentSettings();elements.samples.value='10000';updateSamplingWarning()`);
    assert.equal(run("elements['sampling-warning'].hidden"),false);
    assert.match(run("elements['sampling-warning'].textContent"),/2 seconds per turn/);
    assert.doesNotThrow(()=>run(`gameRequestBody('${opponent}','10000',null,null)`));
    run("elements.samples.value='0';updateSamplingWarning()");
    assert.equal(run("elements['sampling-warning'].hidden"),true);
  }
  run("elements['opponent-choice'].value='mcb';elements.samples.value='400';updateSamplingWarning()");
  assert.equal(run("elements['sampling-warning'].hidden"),true);
  run("elements.samples.value='401';updateSamplingWarning()");
  assert.equal(run("elements['sampling-warning'].hidden"),false);
  run("elements.samples.value='bad';updateSamplingWarning()");
  assert.equal(run("elements['sampling-warning'].hidden"),true);
  run("elements['opponent-choice'].value='random';updateOpponentSettings()");
  assert.equal(run("elements['sampling-warning'].hidden"),true);
  for (const [opponent,fast,slow] of [['mcg',2000,3000],['mcp',11,12]]) {
    run(`elements['opponent-choice'].value='${opponent}';elements.samples.value='${fast}';updateSamplingWarning()`);
    assert.equal(run("elements['sampling-warning'].hidden"),true);
    run(`elements.samples.value='${slow}';updateSamplingWarning()`);
    assert.equal(run("elements['sampling-warning'].hidden"),false);
  }
});
