// ================================================================
// WASM BRIDGE
// ================================================================
let parseReplayFn = null;
let wasmInited = false;

async function initWasm() {
  if (wasmInited) return true;
  
  try {
    const wasmModule = await import('./wasm_parser.js');
    await wasmModule.default(); 
    
    parseReplayFn = wasmModule.parse_replay; 
    setOnnxStatus('ready', 'WASM: READY');
    wasmInited = true;
    return true;
  } catch(e) {
    console.error('Помилка запуску WASM:', e);
    setOnnxStatus('error', 'WASM: CRASH');
    return false;
  }
}

function parseReplayWithWasm(arrayBuffer) {
  if (!parseReplayFn) {
      console.error("Функція парсингу ще не завантажена!");
      return null;
  }
  
  try {
    const bytes = new Uint8Array(arrayBuffer);
    const jsonString = parseReplayFn(bytes);
    return JSON.parse(jsonString); 
  } catch(e) {
    console.error('Помилка парсингу файлу у WASM:', e);
    return null;
  }
}

// ================================================================
// ONNX RUNTIME
// ================================================================
let onnxSession = null;
async function initOnnx() {
  setOnnxStatus('loading', 'ONNX: LOADING');
  try {
    if (typeof ort === 'undefined') {
      await loadScript('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js');
    }

    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';

    const [modelBuffer] = await Promise.all([
      fetch('./ai_coach_v4.onnx').then(r => r.arrayBuffer()),
      initWasm()
    ]);

    onnxSession = await ort.InferenceSession.create(modelBuffer);
    setOnnxStatus('ready', 'ONNX: READY');
    toast('AI Coach модель завантажена ✓');
  } catch(e) {
    setOnnxStatus('error', 'ONNX: NO MODEL');
  }
}

async function runOnnxInference(features) {
  if (!onnxSession) return null;
  try {
    const inputName = onnxSession.inputNames[0];
    const tensor = new ort.Tensor('float32', Float32Array.from(features), [1, features.length]);
    const result = await onnxSession.run({ [inputName]: tensor });
    const probName = onnxSession.outputNames.length > 1 ? onnxSession.outputNames[1] : onnxSession.outputNames[0];
    const probsObj = result[probName];
    if (probsObj.data) {
      return Array.from(probsObj.data).map(Number);
    }
    if (Array.isArray(probsObj) || probsObj.type === 'sequence') {
       const seq = probsObj.data || probsObj;
       const first = seq[0] || seq;
       if (first instanceof Map) return Array.from(first.values()).map(Number);
       if (typeof first === 'object') return Object.values(first).map(Number);
    }
    return null;
  } catch(e) {
    console.warn('ONNX inference error:', e);
    return null;
  }
}

function loadScript(src) {
  return new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = src; s.onload = res; s.onerror = rej;
    document.head.appendChild(s);
  });
}

function setOnnxStatus(state, label) {
  const dot = document.getElementById('onnx-dot');
  const lbl = document.getElementById('onnx-label');
  dot.className = 'status-dot ' + state;
  lbl.textContent = label;
}

// ================================================================
// STATE
// ================================================================
let G = {
  players: [], frames: [], fullDf: {}, hits: [], ballYTimeline: [], boostData: {}, paths: {}, matchInfo: {}, chartInstances: {},
  animState: { playing: false, frame: 0, start: 0, end: 0, rafId: null, targetFrame: 0 },
  fullAnimState: { playing: false, frame: 0, start: 0, end: 0, rafId: null, speed: 1 },
  currentActive3DContainer: 'field-anim-container',
  currentMoment: null, currentAIPredict: null,

  fieldImg: (() => {
    const img = new Image();
    img.onload = () => { if (typeof renderAll === 'function') renderAll(); };
    const b64 = `PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNTg3cHgiIGhlaWdodD0iMjMxNXB4IiB2aWV3Qm94PSIwIDAgMTU4NyAyMzE1Ij48cGF0aCBkPSJNOTI5IDE1ODcuMTlDNzQ2LjA5NyAxNTg3LjE5IDU5Ny44MSAxNDM4LjkgNTk3LjgxIDEyNTZDMjk3LjgxIDEwNzMuMSA3NDYuMDk3IDkyNC44MSA5MjkgOTI0LjgxQzExMTEuOSA5MjQuODEgMTI2MC4xOSAxMDczLjEgMTI2MC4xOSAxMjU2QzEyNjAuMTkgMTM0OC45IDExMTEuOSAxNTg3LjE5IDkyOSAxNTg3LjE5WiIgc3R5bGU9ImZpbGw6bm9uZTtzdHJva2U6I0Y1NzMxNztzdHJva2Utd2lkdGg6NS41NXB4OyIvPjxwYXRoIGQ9Ik04NDEuNDM2IDIzMTQuNTlDNDU4Ljk1NiAyMzE0LjU5IDE0OC44NzggMjAwNC41MSAxNDguODc4IDE2MjIuMDNDMTQ4Ljg3OCAxMjM5LjU1IDE0OC44NzggMTA3NS40NSAxNDguODc4IDY5Mi45NzJDMTQ4Ljg3OCAzMTAuNDkyIDQ1OC45NTYgMC40MTEzMzkgODQxLjQzNiAwLjQxMTMzOUw4NDEuNDM2IDAuNDExMzM5TDg0MS40MzYgMCBMMTU4Ni4zNSAwIEwxNTg2LjM1IDIzMTUgTDg0MS40MzYgMjMxNSBMODQxLjQzNiAyMzE0LjU5TDg0MS40MzYgMjMxNC41OVoiIHN0eWxlPSJmaWxsOiMwRjE1MjA7Ii8+PHBhdGggZD0iTTkyOSAxMjU2TDE1ODYuMzUgMTI1NiIgc3R5bGU9ImZpbGw6bm9uZTtzdHJva2U6I0ZBN0MzMTtzdHJva2Utd2lkdGg6NS43MXB4OyIvPjxyZWN0IHdpZHRoPSIxMjUxLjUzIiBoZWlnaHQ9IjkyOS41NTkiIHg9IjMzNC44MTYiIHk9Ijc5MS4yMjEiIHJ4PSI5MjkuMDE4IiByeT0iOTI5LjAxOCIgc3R5bGU9ImZpbGw6bm9uZTtzdHJva2U6I0ZBN0MzMTtzdHJva2Utd2lkdGg6Mi45MnB4OyIvPjxwYXRoIGQ9Ik0KYjkgOTI0LjgxTDkyOSAwIiBzdHls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`.replace(/\s+/g, '');
    img.src = "data:image/svg+xml;base64," + b64;
    return img;
  })()
};

// ================================================================
// FILE HANDLING
// ================================================================
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', e => { if (e.target.files[0]) handleFile(e.target.files[0]); });

dropZone.addEventListener('click', e => {
  if (e.target.closest('.demo-link') || e.target === fileInput) return;
  fileInput.click();
});

async function handleFile(file) {
  G.replayFileName = file.name.replace(/\.(replay|json)$/i, '');
  showProgress();
  setProgress(5, 'ЧИТАННЯ ФАЙЛУ...');
  const buf = await file.arrayBuffer();

  let data = null;
  if (file.name.endsWith('.json')) {
    setProgress(30, 'ПАРСИНГ JSON...');
    try { data = JSON.parse(new TextDecoder().decode(buf)); } catch(e) {}
  } else if (file.name.endsWith('.replay')) {
    setProgress(20, 'ІНІЦІАЛІЗАЦІЯ WASM ПАРСЕРА...');
    const hasWasm = await initWasm();
    if (hasWasm) {
      setProgress(50, 'ПАРСИНГ .REPLAY...');
      data = parseReplayWithWasm(buf);
    } else {
      hideProgress();
      toast('WASM парсер не знайдено. Використовуй rrrocket локально і завантажуй JSON файл.');
      return;
    }
  }

  if (!data) {
    hideProgress();
    toast('Не вдалось прочитати файл. Спробуй JSON формат.');
    return;
  }

  setProgress(70, 'ОБРОБКА ДАНИХ...');
  await sleep(50);
  await processMatchData(data);

  data = null;

  setProgress(90, 'ONNX AI COACH...');
  await initOnnx();
  await computeAiScores();
  setProgress(100, 'ГОТОВО!');
  await sleep(300);
  showDashboard();
}

// ================================================================
// MATCH DATA PROCESSING
// ================================================================
async function processMatchData(data) {
  const objects = data.objects || [];
  const frames = (data.network_frames || {}).frames || [];
  const props = data.properties || {};

  const teamSize = props.TeamSize || 3;
  const gameMode = `${teamSize}v${teamSize}`;

  const matchTypeStr = props.MatchType || '';
  const isPrivate    = matchTypeStr === 'Private';
  const isRanked     = matchTypeStr === 'Online';
  const isTournament = matchTypeStr === 'LAN' || matchTypeStr === 'Tournament';
  const isCasual     = matchTypeStr === 'Offline' || (!isRanked && !isPrivate && !isTournament);
  
  const MATCH_TYPE_LABELS = { 'Online': 'Ranked', 'Offline': 'Casual', 'Private': 'Private Match', 'LAN': 'LAN', 'Tournament': 'Tournament' };
  const matchType = MATCH_TYPE_LABELS[matchTypeStr] || matchTypeStr || 'Unknown';

  const totalSecondsPlayed = props.TotalSecondsPlayed || 0;
  const team0Goals = props.Team0Score || 0;
  const team1Goals = props.Team1Score || 0;
  
  const timePastRegular = totalSecondsPlayed > 303;
  const goalDiffIsOne = Math.abs(team0Goals - team1Goals) === 1;
  const isOvertime = timePastRegular && goalDiffIsOne;
  const otSecs = isOvertime ? Math.floor(Math.max(0, totalSecondsPlayed - 300)) : 0;

  const replayName = props.ReplayName || props.Id || G.replayFileName || 'Unknown';
  const mapName = props.MapName || '';
  
  let rawStats = props.PlayerStats || [];
  if (!Array.isArray(rawStats)) rawStats = [rawStats];

  G.players = rawStats
    .filter(p => p.Team === 0 || p.Team === 1)
    .map(p => ({
      name: p.Name || 'Unknown', 
      team: p.Team || 0, 
      color: p.Team === 0 ? 'blue' : 'orange',
      score: p.Score || 0, 
      goals: p.Goals || 0, 
      assists: p.Assists || 0,
      saves: p.Saves || 0, 
      shots: p.Shots || 0, 
      demos: p.Demolishes || 0, 
      xg: 0,
      kickoffWins: 0,
      kickoffTouches: 0
    }));

  // ДИНАМІЧНИЙ РЕЗЕРВ (FALLBACK) ЯКЩО PLAYERSTATS ПОРОЖНІЙ (У НОВИХ РЕПЛЕЯХ)
  if (G.players.length === 0) {
    const nameIds = new Set();
    const teamIds = new Set();
    objects.forEach((o, i) => {
      if (o.includes('PlayerName')) nameIds.add(i);
      if (o.includes('PlayerReplicationInfo:Team')) teamIds.add(i);
    });
    
    let tempPlayers = {};
    for (let i = 0; i < frames.length; i++) {
      const fr = frames[i];
      if (!fr || !fr.updated_actors) continue;
      fr.updated_actors.forEach(u => {
        if (nameIds.has(u.object_id) && u.attribute?.String) {
          tempPlayers[u.actor_id] = tempPlayers[u.actor_id] || {};
          tempPlayers[u.actor_id].name = u.attribute.String;
        }
        if (teamIds.has(u.object_id)) {
          const ta = u.attribute?.ActiveActor?.actor ?? u.attribute?.FlaggedInt?.int;
          if (ta != null) {
            tempPlayers[u.actor_id] = tempPlayers[u.actor_id] || {};
            tempPlayers[u.actor_id].teamActor = ta;
          }
        }
      });
    }
    
    // Сортуємо team_actors, щоб присвоїти 0 = Blue, 1 = Orange
    const uniqueTeamActors = [...new Set(Object.values(tempPlayers).map(p => p.teamActor).filter(x => x != null))].sort((a,b) => a - b);
    
    G.players = Object.values(tempPlayers).filter(p => p.name).map(p => {
      let teamNum = uniqueTeamActors.indexOf(p.teamActor);
      if (teamNum === -1) teamNum = 0; // На випадок проблем з Team
      return {
        name: p.name,
        team: teamNum,
        color: teamNum === 0 ? 'blue' : 'orange',
        score: 0, goals: 0, assists: 0, saves: 0, shots: 0, demos: 0, xg: 0, kickoffWins: 0, kickoffTouches: 0
      };
    });
    
    // Дедуплікація на випадок гравців-спостерігачів з тими ж іменами
    const seenNames = new Set();
    G.players = G.players.filter(p => {
      if (seenNames.has(p.name)) return false;
      seenNames.add(p.name);
      return true;
    });
  }

  const actualTeamSize = Math.max(G.players.filter(p => p.color === 'blue').length, G.players.filter(p => p.color === 'orange').length);
  const known = G.players.map(p => p.name);
  G.paths = {}; G.boostData = {};
  G.fullDf = { ball:{x:[],y:[],z:[],vx:[],vy:[],vz:[]} };
  
  known.forEach(n => {
    G.paths[n]={x:[],y:[]}; G.boostData[n]=[];
    G.fullDf[n]={x:[],y:[],z:[],speed:[],boost:[],vx:[],vy:[],vz:[], qx:[],qy:[],qz:[],qw:[]};
  });

  const priToName={}, carToPri={}, compToCar={};
  let ballActorId=null;
  const NF = frames.length;

  const synced = { ball: { 
    x: new Float32Array(NF).fill(NaN), y: new Float32Array(NF).fill(NaN), z: new Float32Array(NF).fill(NaN),
    vx: new Float32Array(NF).fill(NaN), vy: new Float32Array(NF).fill(NaN), vz: new Float32Array(NF).fill(NaN)
  }};
  
  known.forEach(n => { 
    synced[n] = {
      x: new Float32Array(NF).fill(NaN), y: new Float32Array(NF).fill(NaN), z: new Float32Array(NF).fill(NaN),
      speed: new Float32Array(NF).fill(NaN), boost: new Float32Array(NF).fill(NaN),
      vx: new Float32Array(NF).fill(NaN), vy: new Float32Array(NF).fill(NaN), vz: new Float32Array(NF).fill(NaN),
      qx: new Float32Array(NF).fill(NaN), qy: new Float32Array(NF).fill(NaN), qz: new Float32Array(NF).fill(NaN), qw: new Float32Array(NF).fill(NaN)
    }; 
  });

  const knownLower = known.map(n => ({ orig: n, low: n.toLowerCase() }));

  const priObjIds = new Set();
  const vehicleObjIds = new Set();
  
  objects.forEach((o, i) => {
      if (o.includes(':PlayerReplicationInfo') && !o.includes('Default__')) priObjIds.add(i);
      if (o.includes('CarComponent_TA:Vehicle') && !o.includes('Default__')) vehicleObjIds.add(i);
  });

  for (let fi = 0; fi < NF; fi++) {
    const frame = frames[fi];
    if (!frame) continue;

    const deleted_actors = frame.deleted_actors || [];
    for(let i=0; i<deleted_actors.length; i++){
      if(deleted_actors[i] === ballActorId) ballActorId = null;
    }

    const new_actors = frame.new_actors || [];
    for(let i=0; i<new_actors.length; i++){
      const a = new_actors[i];
      if(a.object_id!=null && a.object_id<objects.length) {
        const objName = objects[a.object_id];
        if(objName && objName.includes('Archetypes.Ball.')) {
          ballActorId = a.actor_id;
        }
      }
    }

    const updated_actors = frame.updated_actors || [];
    for(let i=0; i<updated_actors.length; i++){
      const u = updated_actors[i];
      const aid = u.actor_id, oid = u.object_id, attr = u.attribute;
      
      const rawName = attr?.Reservation?.name || attr?.String || null;
      if(rawName && typeof rawName === 'string'){
        const cl = rawName.toLowerCase();
        for(let k=0; k<knownLower.length; k++) {
           if(cl.includes(knownLower[k].low)) priToName[aid] = knownLower[k].orig;
        }
      }
      
      if(priObjIds.has(oid)){ const pa=attr?.ActiveActor?.actor ?? attr?.FlaggedInt?.int; if(pa!=null) carToPri[aid]=pa; }
      if(vehicleObjIds.has(oid)){ const ca=attr?.ActiveActor?.actor ?? attr?.FlaggedInt?.int; if(ca!=null) compToCar[aid]=ca; }
      
      let boostPct=null;
      if(attr?.ReplicatedBoost?.boost_amount!=null) boostPct=Math.round(attr.ReplicatedBoost.boost_amount/255*100);
      else if(attr?.Byte!=null && oid<objects.length && objects[oid].includes('BoostAmount')) boostPct=Math.round(attr.Byte/255*100);
        
      if(boostPct!=null){
        let pn=null;
        if(compToCar[aid]!=null && carToPri[compToCar[aid]]!=null) pn=priToName[carToPri[compToCar[aid]]];
        else if(carToPri[aid]!=null) pn=priToName[carToPri[aid]];
        else pn=priToName[aid];
        if(pn && G.boostData[pn]){ G.boostData[pn].push(boostPct); synced[pn].boost[fi]=boostPct; }
      }
      
      if(attr?.RigidBody){
        const loc = attr.RigidBody.location || {};
        const vel = attr.RigidBody.linear_velocity || {};
        const rot = attr.RigidBody.rotation || {};

        if(aid===ballActorId){
          if(loc.x!=null) synced.ball.x[fi]=loc.x;
          if(loc.y!=null){ synced.ball.y[fi]=loc.y; G.ballYTimeline.push(loc.y); }
          if(loc.z!=null) synced.ball.z[fi]=loc.z;
          if(vel.x!=null) synced.ball.vx[fi]=vel.x;
          if(vel.y!=null) synced.ball.vy[fi]=vel.y;
          if(vel.z!=null) synced.ball.vz[fi]=vel.z;
        }
        
        const pid=carToPri[aid];
        const pn=pid!=null?priToName[pid]:null;
        
        if(pn && known.includes(pn)){
          if(loc.x!=null && loc.y!=null){ 
            G.paths[pn].x.push(loc.x); G.paths[pn].y.push(loc.y); 
            synced[pn].x[fi]=loc.x; synced[pn].y[fi]=loc.y; 
          }
          if(loc.z!=null) synced[pn].z[fi]=loc.z;
          if(vel.x!=null && vel.y!=null && vel.z!=null){
            synced[pn].speed[fi]=Math.sqrt(vel.x**2+vel.y**2+vel.z**2)*0.036;
            synced[pn].vx[fi]=vel.x; synced[pn].vy[fi]=vel.y; synced[pn].vz[fi]=vel.z;
          }
          if (rot.x !== undefined && rot.w !== undefined) {
            synced[pn].qx[fi] = rot.x; synced[pn].qy[fi] = rot.y; synced[pn].qz[fi] = rot.z; synced[pn].qw[fi] = rot.w;
          }
        }
      }
    }
    frames[fi] = null;
    if (fi % 1000 === 0) { await sleep(1); setProgress(70 + Math.floor((fi / NF) * 20), `АНАЛІЗ ТРАЄКТОРІЙ: ${fi} / ${NF}`); }
  }

  function ffillArray(arr, def=0) {
    let last=def;
    for(let i=0; i<arr.length; i++){ if(!Number.isNaN(arr[i])) last = arr[i]; else arr[i] = last; }
    return arr;
  }

  known.forEach(n=>{
    G.fullDf[n].x=ffillArray(synced[n].x, 0); G.fullDf[n].y=ffillArray(synced[n].y, 0); G.fullDf[n].z=ffillArray(synced[n].z, 0);
    G.fullDf[n].speed=ffillArray(synced[n].speed, 0); G.fullDf[n].boost=ffillArray(synced[n].boost, 33);
    G.fullDf[n].vx=ffillArray(synced[n].vx, 0); G.fullDf[n].vy=ffillArray(synced[n].vy, 0); G.fullDf[n].vz=ffillArray(synced[n].vz, 0);
    G.fullDf[n].qx=ffillArray(synced[n].qx, 0); G.fullDf[n].qy=ffillArray(synced[n].qy, 0);
    G.fullDf[n].qz=ffillArray(synced[n].qz, 0); G.fullDf[n].qw=ffillArray(synced[n].qw, 1);
  });

  G.fullDf.ball.x=ffillArray(synced.ball.x, 0); G.fullDf.ball.y=ffillArray(synced.ball.y, 0); G.fullDf.ball.z=ffillArray(synced.ball.z, 0);
  G.fullDf.ball.vx=ffillArray(synced.ball.vx, 0); G.fullDf.ball.vy=ffillArray(synced.ball.vy, 0); G.fullDf.ball.vz=ffillArray(synced.ball.vz, 0);

  // Hits detection
  G.hits=[];
  for(let f=0;f<NF;f++){
      for(let k=0; k<known.length; k++){
        const pn = known[k];
        if (!G.fullDf[pn] || !G.fullDf.ball) continue;
        const px = G.fullDf[pn].x[f], py = G.fullDf[pn].y[f], pz = G.fullDf[pn].z[f];
        const bx = G.fullDf.ball.x[f], by = G.fullDf.ball.y[f], bz = G.fullDf.ball.z[f];
        const d=Math.sqrt((px-bx)**2+(py-by)**2+(pz-bz)**2);
        if(d<300){
          const last=G.hits[G.hits.length-1];
          const p=G.players.find(x=>x.name===pn);
          if(!last || f-last.frame>30){
            const bvx = G.fullDf.ball.vx[f], bvy = G.fullDf.ball.vy[f], bvz = G.fullDf.ball.vz[f];
            const ballSpeed = Math.sqrt(bvx**2 + bvy**2 + bvz**2) * 0.036;
            const isFlick = Math.abs(bx-px) < 200 && Math.abs(by-py) < 200 && (bz-pz) > 50 && (bz-pz) < 300 && ballSpeed > 50;
            const isAerial = pz > 300;
            const targetY = p?.color === 'blue' ? 5120 : -5120;
            let opponentBlocking = false;
            for (let ok = 0; ok < known.length; ok++) {
              const opn = known[ok];
              const op = G.players.find(x => x.name === opn);
              if (!op || op.color === p?.color) continue;
              const ox = G.fullDf[opn].x[f], oy = G.fullDf[opn].y[f];
              const onLine = Math.abs(ox - bx) < 500 && Math.abs(oy - targetY) < Math.abs(by - targetY);
              if (onLine) { opponentBlocking = true; break; }
            }
            G.hits.push({ 
              player: pn, ballX: bx, ballY: by, frame: f, 
              xg: calcXg(bx, by, bz, p?.color || 'blue', ballSpeed, bvy, isFlick, isAerial, opponentBlocking), color: p?.color || 'blue' 
            });
          }
        }
      }
  }

  G.players.forEach(p=>{ p.xg = 0; });
  
  // Позиціювання
  G.players.forEach(p => { p.fClose=0; p.fFar=0; p.sumDist=0; p.fFwd=0; p.fBack=0; p.relXSum=0; p.relYSum=0; p.vFrames=0; });
  for (let f = 0; f < NF; f++) {
    const bx = G.fullDf.ball.x[f], by = G.fullDf.ball.y[f], bz = G.fullDf.ball.z[f];
    ['blue', 'orange'].forEach(team => {
      const members = G.players.filter(p => p.color === team);
      if (!members.length) return;
      let dists = [], ys = [], validMem = [], sumX = 0, sumY = 0;
      members.forEach(p => {
        const px = G.fullDf[p.name].x[f], py = G.fullDf[p.name].y[f], pz = G.fullDf[p.name].z[f];
        if (px !== 0 || py !== 0) { 
          const d = Math.sqrt((px-bx)**2 + (py-by)**2 + (pz-bz)**2);
          dists.push({ p: p, d: d }); ys.push({ p: p, y: py });
          sumX += px; sumY += py; validMem.push(p); p.sumDist += d; p.vFrames++;
        }
      });
      if (dists.length > 0) {
        dists.sort((a, b) => a.d - b.d); dists[0].p.fClose++; dists[dists.length - 1].p.fFar++;
        ys.sort((a, b) => a.y - b.y);
        if (team === 'blue') { ys[ys.length - 1].p.fFwd++; ys[0].p.fBack++; } else { ys[0].p.fFwd++; ys[ys.length - 1].p.fBack++; }
        const cx = sumX / validMem.length, cy = sumY / validMem.length;
        validMem.forEach(p => { p.relXSum += (G.fullDf[p.name].x[f] - cx); p.relYSum += (G.fullDf[p.name].y[f] - cy); });
      }
    });
  }

  G.players.forEach(p => {
    const tot = Math.max(1, p.vFrames);
    p.pctClosest = (p.fClose / tot * 100).toFixed(1) + '%'; p.pctFurthest = (p.fFar / tot * 100).toFixed(1) + '%';
    p.avgDist = Math.round((p.sumDist / tot) / 100) + 'm'; 
    p.pctForward = (p.fFwd / tot * 100).toFixed(1) + '%'; p.pctBack = (p.fBack / tot * 100).toFixed(1) + '%';
    p.relX = p.relXSum / tot; p.relY = p.relYSum / tot;
  });

  G.matchInfo = {
    isOvertime, totalSecs: Math.floor(totalSecondsPlayed || Math.floor(NF/30)),
    blueGoals:  G.players.filter(p=>p.color==='blue').reduce((a,p)=>a+p.goals,0),
    orangeGoals: G.players.filter(p=>p.color==='orange').reduce((a,p)=>a+p.goals,0),
    gameMode: `${actualTeamSize}v${actualTeamSize}`, matchType, isRanked, isTournament, isPrivate, isCasual, otSecs, replayName, mapName,
  };

  G.hits = G.hits.filter((hit, index, self) => {
      if (index === 0) return true;
      const prev = self[index - 1];
      
      // Якщо це той самий гравець і пройшло менше 2 секунд (60 кадрів)
      const tooClose = (hit.frame - prev.frame < 60) && (hit.player === prev.player);
      
      // ігноруємо наступні хіти будь-кого протягом 5 секунд (період святкування)
      const isPostGoalNoise = (prev.isGoal || (prev.action === 3 && prev.xg > 0.5)) && (hit.frame - prev.frame < 150);
      
      return !tooClose && !isPostGoalNoise;
  });

  let filteredHits = [];
  let lastGoalFrame = -500;

  G.hits.forEach(h => {
    // 1. Ігноруємо хіти, якщо нещодавно був гол (пауза 8 секунд)
    if (h.frame < lastGoalFrame + 240) return; 

    // 2. Дедуплікація звичайних ударів (пауза 1 сек)
    const last = filteredHits[filteredHits.length - 1];
    if (last && h.player === last.player && h.frame - last.frame < 30) return;

    if (h.isGoal) lastGoalFrame = h.frame;
    filteredHits.push(h);
  });
  G.hits = filteredHits;

  G.doubleCommits = [];

  for (let f = 0; f < NF; f += 15) { // Перевіряємо кожні 0.5 сек для швидкості
      ['blue', 'orange'].forEach(team => {
          const members = G.players.filter(p => p.color === team);
          if (members.length < 2) return;

          let playersNearBall = [];
          const bx = G.fullDf.ball.x[f], by = G.fullDf.ball.y[f], bz = G.fullDf.ball.z[f];

          members.forEach(p => {
              const px = G.fullDf[p.name].x[f], py = G.fullDf[p.name].y[f], pz = G.fullDf[p.name].z[f];
              const dist = Math.sqrt((px-bx)**2 + (py-by)**2 + (pz-bz)**2);
              const speed = G.fullDf[p.name].speed[f];
              
              let isTowardsBall = false;
              if (dist > 0) {
                  const vx = G.fullDf[p.name].vx[f], vy = G.fullDf[p.name].vy[f], vz = G.fullDf[p.name].vz[f];
                  const vLen = Math.sqrt(vx**2 + vy**2 + vz**2);
                  if (vLen > 0) {
                      const dot = ((bx-px)/dist)*(vx/vLen) + ((by-py)/dist)*(vy/vLen) + ((bz-pz)/dist)*(vz/vLen);
                      isTowardsBall = dot > 0.7; // ~45 градусів (напрямок на м'яч)
                  }
              }
              
              // Якщо гравець близько до м'яча, має швидкість і летить саме на м'яч
              if (dist < 900 && speed > 30 && isTowardsBall) {
                  playersNearBall.push(p.name);
              }
          });

          if (playersNearBall.length >= 2) {
              const lastDC = G.doubleCommits[G.doubleCommits.length - 1];
              // Щоб не спамити, дозволяємо один запис на 3 секунди
              if (!lastDC || f - lastDC.frame > 90) {
                  G.doubleCommits.push({
                      frame: f,
                      team: team,
                      players: [...playersNearBall],
                      ballX: bx, ballY: by
                  });
                  
                  G.hits.push({
                      player: playersNearBall[0],
                      frame: f,
                      action: 0,
                      actionName: 'Double Commit',
                      confidence: 99,
                      isDoubleCommit: true,
                      otherPlayers: playersNearBall.slice(1),
                      xg: 0,
                      ballX: bx, ballY: by, color: team
                  });
              }
          }
      });
  }

  G.players.forEach(p => { p.kickoffWins = 0; p.kickoffTouches = 0; });

  let kickoffFrame = -1;
  for (let f = 0; f < NF; f++) {
      const bx = G.fullDf.ball.x[f], by = G.fullDf.ball.y[f];
      const bvx = G.fullDf.ball.vx[f];
      
      // М'яч у центрі і не рухається - початок кік-оффу
      if (Math.abs(bx) < 1 && Math.abs(by) < 1 && Math.abs(bvx) < 1) {
          kickoffFrame = f;
      }

      // Перший дотик після знаходження м'яча в центрі
      if (kickoffFrame !== -1 && (Math.abs(bvx) > 10)) {
          const firstHitter = G.hits.find(h => h.frame >= f && h.frame < f + 10);
          if (firstHitter) {
              const p = G.players.find(x => x.name === firstHitter.player);
              if (p) {
                  p.kickoffTouches++;
                  // Аналізуємо куди полетів м'яч через 2 сек
                  const futureY = G.fullDf.ball.y[f + 60]; 
                  const win = (p.color === 'blue' && futureY > 1000) || (p.color === 'orange' && futureY < -1000);
                  if (win) p.kickoffWins++;
              }
          }
          kickoffFrame = -1; // Скидаємо до наступного гола
      }
  }
}
// ================================================================
// XG
// ================================================================
function calcXg(x, y, z, team, ballSpeed, bvy, isFlick, isAerial, opponentBlocking) {
  const targetY = team === 'blue' ? 5120 : -5120;
  const inAttackHalf = (team === 'blue' && y > 0) || (team === 'orange' && y < 0);
  if (!inAttackHalf) return 0.01;

  const movingToGoal = (team === 'blue' && bvy > 0) || (team === 'orange' && bvy < 0);
  if (!movingToGoal) return 0.02;

  const dist = Math.sqrt(x**2 + (targetY - y)**2);

  const distFactor = Math.max(0.02, 1 - (dist / 6000));
  const angleFactor = Math.max(0.05, 1 - (Math.abs(x) / 4000));
  const speedBonus = Math.min(0.15, ballSpeed / 400);

  let xg = (distFactor * 0.35 + angleFactor * 0.2) + speedBonus;

  // Висота м'яча — низькі та середні удари небезпечніші за випадкові аеріали
  // Низько (0-150) = звичний удар, середньо (150-400) = небезпечний доплей-шот, високо (400+) = важче точно влучити без навички
  if (z > 100 && z <= 400) {
    xg *= 1.15; // доплей-шоти трохи небезпечніші
  } else if (z > 400) {
    xg *= 0.85; // високі аеріали важче довести до гола без точності
  }

  // Тип удару
  if (isFlick) xg *= 1.25;   // фліки несподівані для вратаря
  if (isAerial) xg *= 1.1;   // аеріальний контакт додає складності для захисту

  // Блокування — якщо опонент на лінії удару, шанс різко падає
  if (opponentBlocking) xg *= 0.4;

  return Math.max(0.01, Math.min(0.6, xg));
}

// ================================================================
// AI SCORES (Гібридна система: ONNX + Rule Engine)
// ================================================================
const ACTION_NAMES = ['Втрата','Пас тімейту','Збір бусту','Удар по воротах','Клір','Бекборд пас','Дриблінг / Контроль','Збереження позиції','Гольова ситуація','Фізичний тиск / Демо','Тіньовий захист','Кік-оф (Kickoff)','Сейв'];
const ACTION_ICONS = ['🚨','🎯','⚡','🚀','🛡️','🏀','🔥','✅','🎯','💥','🥷','⚔️','🧤'];

async function computeAiScores() {
  const useOnnx = !!onnxSession;
  
  // 1. Застосовуємо базову евристику (як запасний варіант, якщо ONNX впаде)
  G.hits.forEach(h => {
    const p = G.players.find(x => x.name === h.player);
    if (!p) return;
    const f = h.frame;
    const bvy = Number(getV(G.fullDf.ball?.vy, f));
    const bvx = Number(getV(G.fullDf.ball?.vx, f));
    const bvz = Number(getV(G.fullDf.ball?.vz, f));
    const bSpeed = Math.sqrt(bvx**2 + bvy**2 + bvz**2) * 0.036;
    applyHeuristic(h, p, bvy, bSpeed);
  });

  // 2. Аналізуємо КОЖЕН хіт через ONNX (не тільки топ-20)
  if (useOnnx) {
    for (let i = 0; i < G.hits.length; i++) {
      const h = G.hits[i];
      const f = h.frame;
      const p = G.players.find(x => x.name === h.player);
      if (!p) continue;
      
      try {
        const features = buildFeatures(h, f, p);
        const probs = await runOnnxInference(features);
        if (probs && probs.length > 0) {
          let bestAction = 0, maxProb = -1;
          probs.forEach((prob, idx) => {
            const pVal = Number(prob); 
            if (pVal > maxProb) { maxProb = pVal; bestAction = idx; }
          });
          if (bestAction >= ACTION_NAMES.length) bestAction = 1;
          
          h.action     = bestAction;
          h.actionName = ACTION_NAMES[bestAction];
          h.confidence = maxProb * 100;
          h.probs      = Array.from(probs);
        }
      } catch(e) {
        console.warn('ONNX skip:', e.message);
      }

      // Оновлюємо прогрес-бар кожні 10 хітів, щоб не гальмувати UI
      if (i % 10 === 0) {
        await new Promise(r => setTimeout(r, 0));
        setProgress(85 + Math.floor((i / G.hits.length) * 13), `AI АНАЛІЗУЄ... ${i+1}/${G.hits.length}`);
      }
    }
  }

  // 3. POST-PROCESSING ENGINE
  G.hits.forEach(h => {
    const p = G.players.find(x => x.name === h.player);
    if (p) applySmartCorrections(h, p);
  });

  // 4. Розрахунок xG
  G.players.forEach(p => {
    const playerHits = G.hits.filter(h => h.player === p.name && h.xg > 0.2);
    
    if (p.shots > 0 && playerHits.length > 0) {
      const topShots = playerHits.sort((a,b) => b.xg - a.xg).slice(0, p.shots);
      const weightedXg = topShots.reduce((sum, h, idx) => {
        const weight = 1 / (1 + idx * 0.35);
        return sum + h.xg * weight;
      }, 0);
      p.xg = Math.min(weightedXg, 1.8);
    } else {
      p.xg = 0;
    }
  });
}

// Допоміжна функція для нормалізації ймовірностей (Soft Scaling)
function boostActionProbability(h, targetAction, boostAmount) {
  // Ініціалізуємо базовий масив на 13 класів, якщо він порожній
  if (!h.probs || h.probs.length === 0) {
    h.probs = new Array(13).fill(0.02);
  }
  
  // Призначаємо нову дію
  h.action = targetAction;
  h.actionName = ACTION_NAMES[targetAction];
  
  // Збільшуємо вагу цільової дії
  h.probs[targetAction] += boostAmount;
  
  // Нормалізуємо масив, щоб сума всіх ймовірностей дорівнювала 1.0
  const sum = h.probs.reduce((a, b) => a + b, 0);
  h.probs = h.probs.map(x => x / sum);
  
  // Оновлюємо confidence на основі нормалізованого значення
  h.confidence = h.probs[targetAction] * 100;
}

function applySmartCorrections(h, p) {
  const f = h.frame;
  const bx = Number(getV(G.fullDf.ball?.x, f));
  const by = Number(getV(G.fullDf.ball?.y, f));
  const bvx = Number(getV(G.fullDf.ball?.vx, f));
  const bvy = Number(getV(G.fullDf.ball?.vy, f));
  const bSpeed = Math.sqrt(bvx**2 + bvy**2 + Number(getV(G.fullDf.ball?.vz,f))**2) * 0.036;

  const ownGoalY = p.color === 'blue' ? -5120 : 5120;
  const oppGoalY = p.color === 'blue' ? 5120 : -5120;
  const distToOwnGoal = Math.abs(by - ownGoalY);
  const toOpponentGoal = (p.color === 'blue' && bvy > 0) || (p.color === 'orange' && bvy < 0);

  // Перевіряємо наступні 150 кадрів (5 секунд), чи перетне м'яч лінію воріт суперника
  let isGoal = false;
  const maxF = Math.min(f + 150, (G.fullDf.ball.y || []).length - 1);

  for (let i = f; i <= maxF; i++) {
    // ПЕРЕВІРКА: чи не було іншого хіта між цим моментом і забиттям гола?
    const intermediateHit = G.hits.find(otherH => otherH.frame > f && otherH.frame < i);
    if (intermediateHit) break; // Якщо хтось інший торкнувся — це був пас, а не прямий гол

    const checkY = Number(getV(G.fullDf.ball.y, i));
    if ((p.color === 'blue' && checkY > 5120) || (p.color === 'orange' && checkY < -5120)) {
      isGoal = true;
      break;
    }
  }

  if (isGoal) {
    h.isGoal = true;
    boostActionProbability(h, 3, 3.0); // Примусово робимо це "Ударом" у предиктах
    h.actionName = 'ГОЛ!'; // Перевизначаємо назву для UI
    return;
  }

  if (Math.abs(bx) < 200 && Math.abs(by) < 200) {
    boostActionProbability(h, 11, 2.5);
    return;
  }

  if (distToOwnGoal < 1500 && bSpeed > 40 && !toOpponentGoal) {
    boostActionProbability(h, 4, 1.5);
    return;
  }

  if (h.action === 3 || h.actionName === 'Удар по воротах') {
    const inAttackingHalf = (p.color === 'blue' && by > 0) || (p.color === 'orange' && by < 0);
    if (!toOpponentGoal || !inAttackingHalf || bSpeed < 30) {
      const newAction = distToOwnGoal < 2500 ? 4 : 1; 
      boostActionProbability(h, newAction, 1.0); 
    }
  }

  if (h.action === 4 || h.action === 12 || h.actionName === 'Клір' || h.actionName === 'Сейв') {
    const distToOppGoal = Math.abs(by - oppGoalY);
    if (distToOppGoal < 2500) {
      const newAction = toOpponentGoal ? 3 : 1;
      boostActionProbability(h, newAction, 1.0);
    }
  }
}

function buildFeatures(h, f, p) {
  const df  = G.fullDf;
  const pn  = h.player;
  const bx  = Number(getV(df.ball?.x,  f));
  const by  = Number(getV(df.ball?.y,  f));
  const bz  = Number(getV(df.ball?.z,  f));
  const px  = Number(getV(df[pn]?.x,   f));
  const py  = Number(getV(df[pn]?.y,   f));
  const pz  = Number(getV(df[pn]?.z,   f));
  const pvx = Number(getV(df[pn]?.vx,  f));
  const pvy = Number(getV(df[pn]?.vy,  f));
  const pvz = Number(getV(df[pn]?.vz,  f));
  const bvx = Number(getV(df.ball?.vx, f));
  const bvy = Number(getV(df.ball?.vy, f));
  const bvz = Number(getV(df.ball?.vz, f));
  const bst = Number(getV(df[pn]?.boost, f));

  const player_speed = Math.sqrt(pvx**2 + pvy**2 + pvz**2);
  const ball_speed   = Math.sqrt(bvx**2 + bvy**2 + bvz**2);
  const horizontal_speed = Math.sqrt(pvx**2 + pvy**2);

  const ownGoalY = p.color === 'blue' ? -5120 : 5120;
  const oppGoalY = p.color === 'blue' ?  5120 : -5120;

  const distOwn     = Math.sqrt(px**2 + (py - ownGoalY)**2);
  const distOpp     = Math.sqrt(px**2 + (py - oppGoalY)**2);
  const distBallOwn = Math.sqrt(bx**2 + (by - ownGoalY)**2);
  const distBallOpp = Math.sqrt(bx**2 + (by - oppGoalY)**2);
  const angleBallOwn = Math.abs(bx) / Math.max(Math.abs(ownGoalY - by), 1);

  let nearTm=9999, nearOpp=9999, oppDistBall=9999;
  let tmAhead=0, tmBehind=0, oppBlocking=0, passingLaneOpen=1;

  G.players.forEach(other => {
    if (other.name === pn) return;
    const ox = Number(getV(df[other.name]?.x, f));
    const oy = Number(getV(df[other.name]?.y, f));
    const oz = Number(getV(df[other.name]?.z, f));
    if (!ox && !oy) return;
    const d     = Math.sqrt((ox-px)**2 + (oy-py)**2);
    const dBall = Math.sqrt((ox-bx)**2 + (oy-by)**2 + (oz-bz)**2);
    if (other.color === p.color) {
      nearTm = Math.min(nearTm, d);
      if ((p.color==='blue' && oy < by) || (p.color==='orange' && oy > by)) tmBehind++;
      if ((p.color==='blue' && oy > by) || (p.color==='orange' && oy < by)) tmAhead++;
    } else {
      nearOpp = Math.min(nearOpp, d);
      oppDistBall = Math.min(oppDistBall, dBall);
      if ((p.color==='blue' && oy < py && Math.abs(ox-px) < 800) ||
          (p.color==='orange' && oy > py && Math.abs(ox-px) < 800)) oppBlocking = 1;
      if (d < nearTm) passingLaneOpen = 0;
    }
  });

  const isLastMan = tmBehind === 0 ? 1 : 0;

  const PADS = [[-3072,-4096],[3072,-4096],[-3584,0],[3584,0],[-3072,4096],[3072,4096],
                [0,-4096],[0,4096],[-1792,-2944],[1792,-2944],[-1792,2944],[1792,2944],
                [-3584,-2560],[3584,-2560],[-3584,2560],[3584,2560]];
  const nearBoostPad = Math.min(...PADS.map(([a,b]) => Math.sqrt((px-a)**2 + (py-b)**2)));
  const ttb_player   = Math.sqrt((px-bx)**2 + (py-by)**2 + (pz-bz)**2) / (player_speed + 1);

  // Плейсхолдери для фіч які недоступні під час інференсу
  const time_remaining = 300;
  const is_overtime    = 0;
  const score_diff     = 0;

  // Механіки
  const is_on_ground    = pz < 120 ? 1 : 0;
  const is_on_wall      = (Math.abs(px) > 3800 || Math.abs(py) > 4900) && pz > 100 ? 1 : 0;
  const is_on_ceiling   = pz > 1900 ? 1 : 0;
  const is_low_aerial   = pz > 120 && pz <= 500 ? 1 : 0;
  const is_mid_aerial   = pz > 500 && pz <= 1200 ? 1 : 0;
  const is_high_aerial  = pz > 1200 ? 1 : 0;

  const ball_is_aerial  = bz > 300;
  const ball_on_wall    = (Math.abs(bx) > 3800 || Math.abs(by) > 4900) && bz > 100;
  const ball_on_ceiling = bz > 1900;
  const ball_directly_above = Math.abs(bx-px) < 200 && Math.abs(by-py) < 200 && (bz-pz) > 50 && (bz-pz) < 300;

  const is_aerial_hit   = is_mid_aerial || is_high_aerial;
  const is_flick        = ball_directly_above && ball_speed > 800 ? 1 : 0;

  const ball_was_going_to_own_goal =
    (p.color === 'blue' && bvy < -800 && by < -3000) ||
    (p.color !== 'blue' && bvy >  800 && by >  3000);
  const is_save      = ball_was_going_to_own_goal && ball_speed > 700 ? 1 : 0;
  const is_wall_save = is_save && is_on_wall ? 1 : 0;

  const is_pinch        = (ball_on_wall || ball_on_ceiling) && ball_speed > 1500 && player_speed > 800 ? 1 : 0;
  const is_ground_pinch = bz < 150 && pz < 150 && pvz < -100 && ball_speed > 2000 ? 1 : 0;
  const is_kuxir_pinch  = is_pinch && Math.abs(bx) > 3600 && Math.abs(by) > 2000 ? 1 : 0;

  const is_air_dribble      = is_aerial_hit && ball_is_aerial && ball_speed < 1200 && player_speed > 600 && Math.abs(bz-pz) < 400 ? 1 : 0;
  const is_fifty_fifty      = nearOpp < 400 && player_speed > 500 && ball_speed > 300 ? 1 : 0;
  const is_supersonic       = player_speed > 2200 ? 1 : 0;
  const is_high_speed_ground = is_on_ground && horizontal_speed > 1800 ? 1 : 0;
  const is_recovery         = is_on_ground && horizontal_speed > 1000 && Math.abs(pvx) > Math.abs(pvy) * 2 ? 1 : 0;
  const is_flip_reset_attempt = is_aerial_hit && Math.abs(bz-pz) < 200 && player_speed > 800 ? 1 : 0;
  const is_psycho           = Math.abs(by - ownGoalY) < 2000 && ball_speed > 1800 &&
                              ((p.color==='blue' && bvy > 800) || (p.color!=='blue' && bvy < -800)) ? 1 : 0;
  const is_fake             = 0;
  const is_redirect         = 0;
  const is_shadowing        = Math.abs(py - ownGoalY) < Math.abs(by - ownGoalY) &&
                              ((p.color==='blue' && pvy < -200 && bvy < -200) ||
                               (p.color!=='blue' && pvy > 200 && bvy > 200)) ? 1 : 0;
  const is_pre_jump         = is_aerial_hit && nearOpp > 1000 && ball_speed > 1000 ? 1 : 0;
  const is_demo_attempt     = is_supersonic && nearOpp < 500 ? 1 : 0;
  const is_boost_starve     = bst > 80 && nearBoostPad < 300 && distOwn > 4000 ? 1 : 0;
  const is_kickoff_touch    = Math.abs(bx) < 400 && Math.abs(by) < 400 && player_speed > 1000 && Math.abs(bz) < 200 ? 1 : 0;

  // 62 елементи — відповідає порядку фіч у датасеті
  return [
    bx, by, bz, px, py, pz,
    player_speed, ball_speed,
    pvx, pvy, bvx, bvy, bvz,
    distOwn, distOpp, distBallOwn, distBallOpp, angleBallOwn,
    nearTm, nearOpp, oppDistBall,
    tmAhead, tmBehind, isLastMan, oppBlocking,
    bst, nearBoostPad,
    ttb_player, passingLaneOpen,
    time_remaining, is_overtime, score_diff,
    is_on_ground, is_on_wall, is_on_ceiling, is_low_aerial, is_mid_aerial, is_high_aerial,
    is_flick, is_save, is_wall_save, is_pinch, is_ground_pinch, is_kuxir_pinch,
    is_air_dribble, is_fifty_fifty, is_supersonic, is_high_speed_ground, is_recovery,
    is_flip_reset_attempt, is_psycho, is_fake, is_redirect, is_shadowing, is_pre_jump,
    is_demo_attempt, is_boost_starve, is_kickoff_touch,
    pz, bz - pz, horizontal_speed, pvz,
  ];
}

function applyHeuristic(h, p, bvy, bSpeed) {
  const inAttack = (p.color==='blue' && h.ballY > 1000) || (p.color==='orange' && h.ballY < -1000);
  const toGoal   = (p.color==='blue' && bvy > 200)      || (p.color==='orange' && bvy < -200);
  
  // bSpeed - це км/год
  const strong   = bSpeed > 45; 

  if (inAttack && toGoal && strong) {
    h.action = 3; h.actionName = 'Удар по воротах';
    h.confidence = 50 + h.xg * 50;
  } else if (!inAttack) {
    h.action = 4;
    h.actionName = 'Клір'; h.confidence = 65 + Math.random() * 25;
  } else {
    h.action = 1;
    h.actionName = 'Пас тімейту'; h.confidence = 55 + Math.random() * 30;
  }

  h.probs = new Array(13).fill(0.02);
  h.probs[h.action] = h.confidence / 100;
  const sum = h.probs.reduce((a,b) => a+b, 0);
  h.probs = h.probs.map(x => x / sum);
} 

function getV(arr, i) {
  const v = (arr && arr[i] != null) ? arr[i] : 0;
  return typeof v === 'bigint' ? Number(v) : (Number(v) || 0);
}

// ================================================================
// ДВИГУН КОНТРФАКТУАЛЬНОГО АНАЛІЗУ
// ================================================================
function calculateAIAlternative(moment) {
  const frame = moment.frame || 0;
  const player = G.players.find(p => p.name === moment.player);
  if (!player) return { advice: "Оцінка неможлива.", prediction: "", ghostTrajectory: null };
  
  const teammates = G.players.filter(p => p.color === player.color && p.name !== player.name);
  const ballX = Number(getV(G.fullDf?.ball?.x, frame)) || 0;
  const ballY = Number(getV(G.fullDf?.ball?.y, frame)) || 0;
  
  const pvx = Number(getV(G.fullDf[player.name]?.vx, frame)) || 0;
  const pvy = Number(getV(G.fullDf[player.name]?.vy, frame)) || 0;
  
  let advice = "Дія розпізнана як тактично правильна в даній позиції.";
  let prediction = "Поточна траєкторія зберегла стабільний темп гри команди.";
  let ghostTrajectory = null;

  const ownGoalY = player.color === 'blue' ? -5120 : 5120;
  const oppGoalY = player.color === 'blue' ? 5120 : -5120;

  const isGoodSave = moment.action === 12 && moment.confidence > 60;
  const isGoodPass = moment.action === 1 && moment.confidence > 60;
  const isGoodClear = moment.action === 4 && Math.abs(ballY) >= 2000;
  const isGoodChallenge = (moment.action === 0 || moment.action === 2 || moment.action === 5) && moment.confidence > 60;
  const isGoodShot = moment.action === 3 && moment.confidence > 50 && !moment.isGoal;

  const isGoodAction = moment.isGoal || isGoodSave || isGoodPass || isGoodClear || isGoodChallenge || isGoodShot;

  // Хвалимо за хороші рішення ===
  if (isGoodAction) {
     if (moment.isGoal) {
         advice = "Блискуча реалізація (Goal Conversion)!";
         prediction = "Точний удар, який не залишив шансів захисту суперника.";
     } else if (isGoodSave) {
         advice = "Ключовий Сейв (Clutch Save)!";
         prediction = "Вчасна реакція врятувала команду від неминучого голу.";
     } else if (isGoodPass) {
         advice = "Чудове бачення поля (Smart Pass)!";
         prediction = "Точний пас зберіг темп атаки і розірвав захист суперників.";
     } else if (isGoodClear) {
         advice = "Ефективний Виніс (Deep Clear)!";
         prediction = "М'яч вибито в безпечну зону, що зняло тиск і дало команді час на ротацію.";
     } else if (isGoodShot) {
         advice = "Небезпечний Удар (High xG Shot)!";
         prediction = "Хороша спроба загострити гру, яка змусила суперника витратити ресурси.";
     } else {
         advice = "Впевнений Челендж / Перехоплення (Smart Play)!";
         prediction = "Виграна боротьба або вчасна агресія забезпечили контроль над ситуацією.";
     }

     let contextNotes = [];
     const lookbackFrames = 150; // ~5 seconds
     
     // 1. Асист (попередній дотик тімейта)
     const recentHits = G.hits.filter(h => h.frame < frame && h.frame > frame - lookbackFrames);
     const prevHit = recentHits.reverse().find(h => h.color === player.color && h.player !== player.name);
     if (prevHit) {
         contextNotes.push(`Своєчасна дія від <b>${prevHit.player}</b> допомогла розірвати позицію суперників.`);
     }

     // 2. Помилка суперників (Double Commit)
     const oppTeam = player.color === 'blue' ? 'orange' : 'blue';
     const recentDC = G.doubleCommits.find(dc => dc.team === oppTeam && dc.frame < frame && dc.frame > frame - lookbackFrames);
     if (recentDC) {
         contextNotes.push(`Суперники зробили <b>Double Commit</b>, залишивши зону відкритою.`);
     }

     // 3. Перевага по бусту
     const opponents = G.players.filter(p => p.color === oppTeam);
     const lowBoostOpps = opponents.filter(opp => getV(G.fullDf[opp.name]?.boost, Math.max(0, frame - 30)) < 20);
     if (lowBoostOpps.length > 0) {
         contextNotes.push(`Суперники (${lowBoostOpps.map(o=>`<b>${o.name}</b>`).join(', ')}) були виснажені (низький буст).`);
     }
     
     // 4. Швидка контратака
     const ballYPast = getV(G.fullDf?.ball?.y, Math.max(0, frame - 150));
     const yDiff = ballY - ballYPast;
     if ((player.color === 'blue' && yDiff > 4000) || (player.color === 'orange' && yDiff < -4000)) {
         contextNotes.push(`Миттєва контратака (Fast Transition) не дала захисту часу повернутись.`);
     }

     if (contextNotes.length > 0) {
         prediction += "<br><br><span style='color:var(--green);font-size:12px'><b>ЩО ПРИЗВЕЛО ДО УСПІХУ:</b></span><br>• " + contextNotes.join("<br>• ");
     }

     ghostTrajectory = null; // Не показуємо "альтернативу", бо дія вже ідеальна
  }
  // ============================
  else if (moment.action === 0) { 
    let bestTm = null; let maxDist = -1;
    teammates.forEach(t => {
       const tx = Number(getV(G.fullDf[t.name]?.x, frame));
       const ty = Number(getV(G.fullDf[t.name]?.y, frame));
       const d = Math.sqrt((tx-ballX)**2 + (ty-ballY)**2);
       if (d > maxDist) { maxDist = d; bestTm = {name: t.name, x: tx, y: ty}; }
    });
    
    if (bestTm && maxDist > 1200) {
       advice = `Уникнення втрати (Possession Retention). Оптимально: пас назад/вбік на гравця ${bestTm.name}.`;
       prediction = "Це дозволило б зберегти контроль над м'ячем і безпечно вийти з-під пресингу суперника.";
       ghostTrajectory = { fromX: ballX, fromY: ballY, toX: bestTm.x, toY: bestTm.y, pvx, pvy };
    } else {
       advice = "Невиправданий челендж. Оптимально: Shadow Defense (Тіньовий захист).";
       prediction = "Відкат назад із фейковим випадом виграв би 2.5с для повернення тімейтів у ротацію.";
       ghostTrajectory = { fromX: ballX, fromY: ballY, toX: ballX * 0.5, toY: ownGoalY * 0.7, pvx, pvy };
    }
  } 
  else if (moment.action === 3 && moment.confidence < 50) { 
    const openTeammate = teammates.find(t => {
      const tY = Number(getV(G.fullDf[t.name]?.y, frame));
      return player.color === 'blue' ? (tY > ballY + 500) : (tY < ballY - 500);
    });
    if (openTeammate) {
      const tX = Number(getV(G.fullDf[openTeammate.name]?.x, frame));
      const tY = Number(getV(G.fullDf[openTeammate.name]?.y, frame));
      advice = `Низький шанс на гол (Low xG). Оптимально: Передача на хід гравцю ${openTeammate.name} (Infield Pass).`;
      prediction = `Пас на дальню штангу повністю відрізав би воротаря та дестабілізував би оборону суперника.`;
      ghostTrajectory = { fromX: ballX, fromY: ballY, toX: tX, toY: tY, pvx, pvy };
    } else {
      advice = "Замало шансів забити прямим ударом. Оптимально: Гра через борт або Бекборд пас.";
      prediction = "Скидання м'яча в щит над воротами змусило б суперників витратити буст і зламало б їхню ротацію.";
      const cornerX = ballX > 0 ? 3000 : -3000;
      ghostTrajectory = { fromX: ballX, fromY: ballY, toX: cornerX, toY: oppGoalY, pvx, pvy };
    }
  } 
  else if (moment.action === 4 && Math.abs(ballY) < 2000) { 
    advice = "Непродуктивне скидання м'яча (Panic Clear). Оптимально: Catch & Carry.";
    prediction = "М'який прийом м'яча та вихід через дриблінг змусили б першого захисника зробити ранній челендж.";
    const wallX = ballX > 0 ? 3800 : -3800;
    ghostTrajectory = { fromX: ballX, fromY: ballY, toX: wallX, toY: ballY + (player.color === 'blue' ? 1000 : -1000), pvx, pvy };
  }

  if (!ghostTrajectory && !moment.isGoal) {
     const pushY = player.color === 'blue' ? ballY + 2500 : ballY - 2500;
     ghostTrajectory = { fromX: ballX, fromY: ballY, toX: ballX + (pvx/Math.max(1, Math.abs(pvx)))*500, toY: pushY, pvx, pvy };
  }

  return { advice, prediction, ghostTrajectory };
}

// ================================================================
// AI COACH UI
// ================================================================
function renderMomentChips() {
  const container = document.getElementById('moment-chips');
  container.innerHTML = '';

  // 1. Фільтруємо важливі моменти
  const goals = G.hits.filter(h => h.isGoal);
  const dcs = G.hits.filter(h => h.isDoubleCommit);
  const shots = G.hits.filter(h => h.action === 3 && !h.isGoal).sort((a,b) => b.xg - a.xg).slice(0, 5);
  const saves = G.hits.filter(h => h.action === 12).slice(0, 3);
  const passes = G.hits.filter(h => h.action === 1).slice(0, 3);
  const clears = G.hits.filter(h => h.action === 4).slice(0, 3);

  // 2. Об'єднуємо все в один список і сортуємо за часом матчу
  const allMoments = [...goals, ...dcs, ...shots, ...saves, ...passes, ...clears].sort((a, b) => a.frame - b.frame);
  
  allMoments.forEach((h) => {
    const timeLabel = formatGameTime(h.frame); 
    const chip = document.createElement('button');
    
    // Встановлюємо стиль: якщо це дабл-коміт — робимо його червоним
    if (h.isDoubleCommit) {
      chip.className = 'moment-chip bad';
      chip.textContent = `🚨 Double Commit ${timeLabel} ${h.player}`;
    } else if (h.isGoal) {
      chip.className = 'moment-chip good';
      chip.textContent = `⚽ ГОЛ! ${timeLabel} ${h.player}`;
    } else {
      chip.className = 'moment-chip';
      chip.textContent = `${ACTION_ICONS[h.action] || '▪'} ${h.actionName} ${timeLabel}`;
    }
    
    chip.onclick = () => selectMoment(h, chip);
    container.appendChild(chip);
  });
}

function selectMoment(hit, chipEl) {
  document.querySelectorAll('.moment-chip').forEach(c=>c.classList.remove('active'));
  chipEl.classList.add('active');
  G.currentMoment = hit;
  
  // 1. Активуємо контейнер і змушуємо Three.js оновитися
  document.getElementById('ai-analysis').style.display='block';
  G.currentActive3DContainer = 'field-anim-container';
  ensure3DScene('field-anim-container'); 
  
  // 2. Оновлюємо розміри рендерера примусово
  const container = document.getElementById('field-anim-container');
  renderer3D.setSize(container.clientWidth, container.clientHeight);
  camera3D.aspect = container.clientWidth / container.clientHeight;
  camera3D.updateProjectionMatrix();

  renderAiMoment(hit);
}

function renderAiMoment(hit) {
  const vbox = document.getElementById('ai-verdict-box');
  const predictPanel = document.getElementById('ai-predict-panel');
  
  if (hit.isDoubleCommit) {
    vbox.className = 'ai-verdict bad';
    vbox.innerHTML = `🚨 <strong>DOUBLE COMMIT</strong> — ${hit.player} та ${hit.otherPlayers.join(', ')} пішли одночасно!`;
    
    // Спеціальний опис для Double Commit
    const analysis = {
        advice: "Злам ротації: хтось мав залишитися в захисті.",
        prediction: "Це залишає ваші ворота порожніми на 3-4 секунди. Один з вас повинен був розвернутися назад.",
        ghostTrajectory: null
    };
    
    // Відображаємо альтернативу для помилки
    G.currentAIPredict = null; 
    let predictPanel = document.getElementById('ai-predict-panel');
    if (predictPanel) {
      predictPanel.innerHTML = `
        <div style="color:#ef4444; font-size:12px; font-weight:700; margin-bottom:5px;">🚨 ТАКТИЧНА ПОМИЛКА</div>
        <div style="color:#f8fafc; font-size:14px; line-height: 1.4;"><b>Порада:</b> Злам ротації. Один з вас мав залишитися "Last Man".</div>
      `;
    }
    
    document.getElementById('ai-probs').innerHTML = '';
    document.getElementById('ai-outcome').innerHTML = '<span style="color:var(--red)">Втрата позиції</span>';
  } else {
    // ЛОГІКА ДЛЯ ЗВИЧАЙНИХ ХІТІВ
    const isGood = hit.action !== 0;
    vbox.className = 'ai-verdict ' + (isGood ? 'good' : 'bad');
    vbox.innerHTML = `${ACTION_ICONS[hit.action] || '▪'} <strong>${hit.actionName || 'Момент'}</strong> — впевненість: ${(hit.confidence || 60).toFixed(0)}%`;

    const probs = hit.probs || [];
    const topIndices = probs
      .map((p, i) => ({ p, i }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 3)
      .map(x => x.i);

    const probsHtml = topIndices.map(i => {
      const p = probs[i];
      return `
        <div class="ai-bar-item">
          <span class="ai-bar-label">${ACTION_ICONS[i]} ${ACTION_NAMES[i]}</span>
          <div class="ai-bar-track"><div class="ai-bar-fill" style="width:${(p * 100).toFixed(0)}%;background:${i === hit.action ? 'var(--accent)' : 'var(--muted)'}"></div></div>
          <span class="ai-bar-val">${(p * 100).toFixed(0)}%</span>
        </div>`;
    }).join('');
    document.getElementById('ai-probs').innerHTML = probsHtml;

    // Результат удару
    const f = hit.frame;
    const AFTER = 90;
    const fNext = Math.min(f + AFTER, (G.fullDf.ball.y || []).length - 1);
    const byNow = getV(G.fullDf.ball.y, f);
    const byAfter = getV(G.fullDf.ball.y, fNext);
    const delta = byAfter - byNow;
    const pc = hit.color;
    let outcomeHtml = '';
    if ((pc === 'blue' && byAfter > 5050) || (pc === 'orange' && byAfter < -5050))
      outcomeHtml = '<span style="color:var(--green)">🎉 ГОЛ! Удар досяг мети!</span>';
    else if ((pc === 'blue' && delta > 800) || (pc === 'orange' && delta < -800))
      outcomeHtml = '<span style="color:var(--blue)">🚀 М\'яч пішов в атаку — хороший тиск!</span>';
    else if ((pc === 'blue' && delta < -800) || (pc === 'orange' && delta > 800))
      outcomeHtml = '<span style="color:var(--red)">💀 М\'яч пішов у твої ворота!</span>';
    else
      outcomeHtml = '<span style="color:var(--muted)">🔄 Позиція стабілізувалась.</span>';
    document.getElementById('ai-outcome').innerHTML = outcomeHtml;

    const analysis = calculateAIAlternative(hit);
    G.currentAIPredict = analysis.ghostTrajectory;

    let predictPanel = document.getElementById('ai-predict-panel');
    if (predictPanel) {
      predictPanel.innerHTML = `
        <div style="color:#22c55e; font-size:12px; font-weight:700; margin-bottom:5px;">💡 AI OPTIMAL ALTERNATIVE</div>
        <div style="color:#f8fafc; font-size:14px; margin-bottom:5px; line-height: 1.4;"><b>Дія:</b> ${analysis.advice}</div>
        <div style="color:#94a3b8; font-size:13px; line-height: 1.4; border-top: 1px solid var(--border); padding-top: 6px;"><b>Прогноз:</b> ${analysis.prediction}</div>
      `;
    }
  }

  // Спільна частина для всіх типів моментів
  const f = hit.frame;
  const pn = hit.player;
  const boost = getV(G.fullDf[pn]?.boost, f);
  const pspeed = getV(G.fullDf[pn]?.speed, f);
  const bvx = getV(G.fullDf.ball.vx, f);
  const bvy = getV(G.fullDf.ball.vy, f);
  const bvz = getV(G.fullDf.ball.vz, f);
  const bspeed = Math.sqrt(bvx**2 + bvy**2 + bvz**2) * 0.036;

  document.getElementById('sensor-grid').innerHTML = `
    <div class="sensor-item"><div class="sensor-key">Буст</div><div class="sensor-val" style="color:${boost < 20 ? 'var(--red)' : 'var(--green)'}">${boost.toFixed(0)}%</div></div>
    <div class="sensor-item"><div class="sensor-key">Швидкість гравця</div><div class="sensor-val">${pspeed.toFixed(0)} <span style="font-size:11px;color:var(--muted)">км/год</span></div></div>
    <div class="sensor-item"><div class="sensor-key">Швидкість м'яча</div><div class="sensor-val">${bspeed.toFixed(0)} <span style="font-size:11px;color:var(--muted)">км/год</span></div></div>
    <div class="sensor-item"><div class="sensor-key">xG</div><div class="sensor-val" style="color:var(--yellow)">${(hit.xg || 0).toFixed(3)}</div></div>
  `;

  startAnim(hit);
}

// ================================================================
// THREE.JS 3D SCENE (Ізометричний стиль)
// ================================================================
let scene3D, camera3D, renderer3D, controls3D;
let ball3D, playerMeshes3D = {};
let playerNameplates = {};
let fieldGroup3D;

function initThreeScene(containerId = 'field-anim-container') {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = '';
  playerNameplates = {}; 
  playerMeshes3D = {}; 

  scene3D = new THREE.Scene();
  scene3D.background = new THREE.Color(0x0a0e16);

  const W = container.clientWidth || 280;
  const H = container.clientHeight || 400;

  camera3D = new THREE.PerspectiveCamera(25, W / H, 1, 50000);

  renderer3D = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer3D.setSize(W, H);
  
  renderer3D.shadowMap.enabled = true;
  renderer3D.shadowMap.type = THREE.PCFSoftShadowMap; 
  
  container.appendChild(renderer3D.domElement);

  const uiLayer = document.createElement('div');
  uiLayer.id = 'anim-ui-layer';
  uiLayer.style.position = 'absolute'; uiLayer.style.top = '0'; uiLayer.style.left = '0';
  uiLayer.style.width = '100%'; uiLayer.style.height = '100%';
  uiLayer.style.pointerEvents = 'none'; 
  container.appendChild(uiLayer);

  const boostPanel = document.createElement('div');
  boostPanel.style.position = 'absolute'; boostPanel.style.top = '16px'; boostPanel.style.left = '16px'; boostPanel.style.right = '16px';
  boostPanel.style.display = 'flex'; boostPanel.style.justifyContent = 'space-between';
  uiLayer.appendChild(boostPanel);

  const blueBoosts = document.createElement('div');
  blueBoosts.id = 'ui-blue-boosts';
  blueBoosts.style.display = 'flex'; blueBoosts.style.flexDirection = 'column'; blueBoosts.style.gap = '6px';
  
  const orangeBoosts = document.createElement('div');
  orangeBoosts.id = 'ui-orange-boosts';
  orangeBoosts.style.display = 'flex'; orangeBoosts.style.flexDirection = 'column'; orangeBoosts.style.gap = '6px'; orangeBoosts.style.alignItems = 'flex-end';
  
  boostPanel.appendChild(blueBoosts);
  boostPanel.appendChild(orangeBoosts);

  const nameplatesDiv = document.createElement('div');
  nameplatesDiv.id = 'ui-nameplates';
  uiLayer.appendChild(nameplatesDiv);

  const ambient = new THREE.AmbientLight(0xffffff, 0.75);
  scene3D.add(ambient);
  
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
  dirLight.position.set(5000, 10000, 5000);
  dirLight.castShadow = true; 
  
  const dArea = 7000;
  dirLight.shadow.camera.left = -dArea;
  dirLight.shadow.camera.right = dArea;
  dirLight.shadow.camera.top = dArea;
  dirLight.shadow.camera.bottom = -dArea;
  dirLight.shadow.camera.near = 500;
  dirLight.shadow.camera.far = 25000;
  dirLight.shadow.mapSize.width = 2048; 
  dirLight.shadow.mapSize.height = 2048;
  
  scene3D.add(dirLight);

  fieldGroup3D = new THREE.Group();

  const fieldW = 4096; const fieldL = 5120;
  const corner = 1152; const wallH = 2000;

  function createHalf(isBlue) {
    const shape = new THREE.Shape();
    const dir = isBlue ? 1 : -1; 
    shape.moveTo(-fieldW, 0); shape.lineTo(fieldW, 0);
    shape.lineTo(fieldW, (fieldL - corner) * dir); shape.lineTo(fieldW - corner, fieldL * dir);
    shape.lineTo(-fieldW + corner, fieldL * dir); shape.lineTo(-fieldW, (fieldL - corner) * dir);
    shape.lineTo(-fieldW, 0);
    const geo = new THREE.ShapeGeometry(shape);
    const mat = new THREE.MeshBasicMaterial({ color: isBlue ? 0x3b82f6 : 0xf97316, transparent: true, opacity: 0.85, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.receiveShadow = true;
    return mesh;
  }

  fieldGroup3D.add(createHalf(true)); 
  fieldGroup3D.add(createHalf(false)); 

  function createWall(x1, z1, x2, z2, color) {
    const dx = x2 - x1, dz = z2 - z1;
    const len = Math.sqrt(dx * dx + dz * dz);
    const geo = new THREE.PlaneGeometry(len, wallH);
    const mat = new THREE.MeshBasicMaterial({ color: color, transparent: true, opacity: 0.15, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set((x1 + x2) / 2, wallH / 2, (z1 + z2) / 2);
    mesh.rotation.y = -Math.atan2(dz, dx);
    const edges = new THREE.EdgesGeometry(geo);
    const lineMat = new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.4 });
    mesh.add(new THREE.LineSegments(edges, lineMat));
    return mesh;
  }

  const cBlue = 0x60a5fa;
  fieldGroup3D.add(createWall(-fieldW, 0, -fieldW, -fieldL+corner, cBlue));
  fieldGroup3D.add(createWall(-fieldW, -fieldL+corner, -fieldW+corner, -fieldL, cBlue));
  fieldGroup3D.add(createWall(-fieldW+corner, -fieldL, fieldW-corner, -fieldL, cBlue)); 
  fieldGroup3D.add(createWall(fieldW-corner, -fieldL, fieldW, -fieldL+corner, cBlue));
  fieldGroup3D.add(createWall(fieldW, -fieldL+corner, fieldW, 0, cBlue));

  const cOrg = 0xfb923c;
  fieldGroup3D.add(createWall(-fieldW, 0, -fieldW, fieldL-corner, cOrg));
  fieldGroup3D.add(createWall(-fieldW, fieldL-corner, -fieldW+corner, fieldL, cOrg));
  fieldGroup3D.add(createWall(-fieldW+corner, fieldL, fieldW-corner, fieldL, cOrg)); 
  fieldGroup3D.add(createWall(fieldW-corner, fieldL, fieldW, fieldL-corner, cOrg));
  fieldGroup3D.add(createWall(fieldW, fieldL-corner, fieldW, 0, cOrg));

  const lineGeo = new THREE.PlaneGeometry(fieldW * 2, 40);
  const lineMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.3 });
  const centerLine = new THREE.Mesh(lineGeo, lineMat);
  centerLine.rotation.x = -Math.PI / 2; centerLine.position.y = 5;
  centerLine.receiveShadow = true; 
  fieldGroup3D.add(centerLine);

  const ringGeo = new THREE.RingGeometry(900, 940, 64);
  const ringMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.3, side: THREE.DoubleSide });
  const centerRing = new THREE.Mesh(ringGeo, ringMat);
  centerRing.rotation.x = -Math.PI / 2; centerRing.position.y = 5;
  centerRing.receiveShadow = true;
  fieldGroup3D.add(centerRing);

  // === БУСТ ПАДИ (Великі і Малі) ===
  const largeBoosts = [[-3072, -4096], [3072, -4096], [-3584, 0], [3584, 0], [-3072, 4096], [3072, 4096]];
  const smallBoosts = [
    [0,-4240], [0,4240], [-1792,-4184], [1792,-4184], [-1792,4184], [1792,4184], 
    [-940,-3308], [940,-3308], [-940,3308], [940,3308], [-1792,-1448], [1792,-1448], 
    [-1792,1448], [1792,1448], [-2560,-2816], [2560,-2816], [-2560,2816], [2560,2816], 
    [0,-2816], [0,2816], [-3584,-1448], [3584,-1448], [-3584,1448], [3584,1448], 
    [-1024,0], [1024,0], [0,-1024], [0,1024]
  ];

  const boostMat = new THREE.MeshStandardMaterial({ 
      color: 0xfacc15, emissive: 0xfacc15, emissiveIntensity: 0.8, transparent: true, opacity: 0.9 
  });

  // Малюємо великі бусти
  const lgGeo = new THREE.SphereGeometry(120, 16, 16);
  largeBoosts.forEach(pos => {
      const m = new THREE.Mesh(lgGeo, boostMat);
      m.position.set(pos[0], 120, pos[1]); // pos[1] - це вісь Z у Three.js
      m.castShadow = true;
      fieldGroup3D.add(m);
      
      const ring = new THREE.Mesh(new THREE.RingGeometry(150, 180, 16), new THREE.MeshBasicMaterial({color: 0xfacc15, transparent: true, opacity: 0.5, side: THREE.DoubleSide}));
      ring.rotation.x = -Math.PI / 2; ring.position.set(pos[0], 5, pos[1]);
      fieldGroup3D.add(ring);
  });

  // Малюємо малі падики
  const smGeo = new THREE.CylinderGeometry(50, 50, 15, 16);
  smallBoosts.forEach(pos => {
      const m = new THREE.Mesh(smGeo, boostMat);
      m.position.set(pos[0], 10, pos[1]);
      m.receiveShadow = true;
      fieldGroup3D.add(m);
  });
  // ===================================

  function createGoal(zPos, isBlue) {
    const gGeo = new THREE.BoxGeometry(1786, 642, 880);
    const gMat = new THREE.MeshBasicMaterial({ color: isBlue ? 0x3b82f6 : 0xf97316, transparent: true, opacity: 0.15, depthWrite: false });
    const goal = new THREE.Mesh(gGeo, gMat);
    goal.position.set(0, 321, zPos);
    const edges = new THREE.EdgesGeometry(gGeo);
    const lineMat = new THREE.LineBasicMaterial({ color: isBlue ? 0x60a5fa : 0xfb923c, transparent: true, opacity: 0.6 });
    goal.add(new THREE.LineSegments(edges, lineMat));
    return goal;
  }
  
  fieldGroup3D.add(createGoal(-fieldL - 440, true));
  fieldGroup3D.add(createGoal(fieldL + 440, false));
  scene3D.add(fieldGroup3D);

  const ballGeo = new THREE.SphereGeometry(100, 32, 32);
  const ballMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.2 });
  ball3D = new THREE.Mesh(ballGeo, ballMat);
  ball3D.castShadow = true; 
  scene3D.add(ball3D);

  setupSimpleOrbit(container);

}

function ensure3DScene(containerId) {
  if (!scene3D) {
    initThreeScene(containerId);
  } else {
    const target = document.getElementById(containerId);
    if (target && renderer3D && renderer3D.domElement.parentElement !== target) {
      target.innerHTML = '';
      target.appendChild(renderer3D.domElement);
      const uiLayer = document.getElementById('anim-ui-layer');
      if (uiLayer) target.appendChild(uiLayer);
      
      const W = target.clientWidth || 280;
      const H = target.clientHeight || 400;
      renderer3D.setSize(W, H);
      camera3D.aspect = W / H;
      camera3D.updateProjectionMatrix();
    }
  }
}

function setupSimpleOrbit(container) {
  const el = renderer3D.domElement;
  let isDragging = false, prevX = 0, prevY = 0;
  
  let theta = Math.PI / 4; 
  let phi = Math.PI / 3;   
  let radius = 29000;

  function updateCamera() {
    camera3D.position.x = radius * Math.sin(phi) * Math.sin(theta);
    camera3D.position.z = radius * Math.sin(phi) * Math.cos(theta);
    camera3D.position.y = radius * Math.cos(phi);
    camera3D.lookAt(0, 0, 0);

    if (G.animState && !G.animState.playing && G.currentMoment) {
      drawAnimFrame3D(G.animState.frame, G.currentMoment);
    } else if (renderer3D && scene3D && camera3D) {
      renderer3D.render(scene3D, camera3D);
    }
  }
  
  updateCamera();

  el.addEventListener('mousedown', e => { 
    isDragging = true; 
    prevX = e.clientX; 
    prevY = e.clientY; 
  });
  
  el.addEventListener('mouseup', () => isDragging = false);
  el.addEventListener('mouseleave', () => isDragging = false);
  
  el.addEventListener('mousemove', e => {
    if (!isDragging) return;
    const dx = e.clientX - prevX, dy = e.clientY - prevY;
    theta -= dx * 0.005;
    phi = Math.max(0.1, Math.min(Math.PI/2 - 0.1, phi - dy * 0.005));
    prevX = e.clientX; 
    prevY = e.clientY;
    updateCamera();
  });
  
  el.addEventListener('wheel', e => {
    e.preventDefault();
    radius = Math.max(8000, Math.min(35000, radius + e.deltaY * 15));
    updateCamera();
  }, { passive: false });
}

// ================================================================
// Хелпер для плавного переходу (Lerp) між кадрами
// ================================================================
function lerpVal(arr, fiFloat) {
  if (!arr || arr.length === 0) return 0;
  const f0 = Math.floor(fiFloat);
  const f1 = Math.ceil(fiFloat);
  const t = fiFloat - f0;
  const getVal = (index) => typeof getV === 'function' ? getV(arr, index) : (arr[index] || 0);
  const v0 = getVal(f0);
  if (f0 === f1) return v0;
  const v1 = getVal(Math.min(f1, arr.length - 1));
  return v0 + (v1 - v0) * t;
}

// ================================================================
// СТВОРЕННЯ МОДЕЛЕЙ
// ================================================================
function getOrCreatePlayerMesh(p) {
  if (playerMeshes3D[p.name]) return playerMeshes3D[p.name];
  
  const isBlue = p.color === 'blue';
  const mainColor = isBlue ? 0x3b82f6 : 0xf97316;

  const carGroup = new THREE.Group();
  
  const bodyGeo = new THREE.BoxGeometry(150, 45, 250);
  const bodyMat = new THREE.MeshStandardMaterial({ 
    color: mainColor, roughness: 0.2, metalness: 0.3,
    emissive: mainColor, emissiveIntensity: 0.15 
  });
  const body = new THREE.Mesh(bodyGeo, bodyMat);
  body.position.y = 22.5; 
  body.castShadow = true;
  carGroup.add(body);
  carGroup.userData.bodyMesh = body; 

  const edges = new THREE.EdgesGeometry(bodyGeo);
  const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 });
  body.add(new THREE.LineSegments(edges, lineMat));

  const roofGeo = new THREE.BoxGeometry(110, 30, 110);
  const roofMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.1, metalness: 0.8 });
  const roof = new THREE.Mesh(roofGeo, roofMat);
  roof.position.set(0, 58, -25); 
  roof.castShadow = true;
  carGroup.add(roof);
  roof.add(new THREE.LineSegments(new THREE.EdgesGeometry(roofGeo), lineMat));

  const flameGeo = new THREE.ConeGeometry(30, 90, 8);
  const flameMat = new THREE.MeshBasicMaterial({ color: 0xfacc15, transparent: true, opacity: 0.9 });
  const flame = new THREE.Mesh(flameGeo, flameMat);
  flame.rotation.x = Math.PI / 2; 
  flame.position.set(0, 22.5, -160);
  flame.visible = false; 
  carGroup.userData.flame = flame; 
  carGroup.add(flame);

  const trailMeshes = [];
  const trailGeo = new THREE.BoxGeometry(25, 25, 25);
  const trailMatObj = new THREE.MeshBasicMaterial({ color: 0xfacc15, transparent: true, opacity: 0.6 });
  for (let i = 0; i < 18; i++) {
    const tMesh = new THREE.Mesh(trailGeo, trailMatObj.clone());
    tMesh.visible = false;
    scene3D.add(tMesh);
    trailMeshes.push(tMesh);
  }
  carGroup.userData.trailMeshes = trailMeshes;

  const teamRingGeo = new THREE.RingGeometry(160, 180, 32);
  const teamRingMat = new THREE.MeshBasicMaterial({ color: mainColor, transparent: true, opacity: 0.5, side: THREE.DoubleSide });
  const teamRing = new THREE.Mesh(teamRingGeo, teamRingMat);
  teamRing.rotation.x = -Math.PI / 2;
  teamRing.position.y = 1; 
  carGroup.add(teamRing);

  const ringGeo = new THREE.RingGeometry(210, 250, 32);
  const ringMat = new THREE.MeshBasicMaterial({ color: 0xeab308, transparent: true, opacity: 0, side: THREE.DoubleSide });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 2; 
  carGroup.userData.hitRing = ring;
  carGroup.add(ring);

  scene3D.add(carGroup);
  playerMeshes3D[p.name] = carGroup;
  return carGroup;
}

// ================================================================
// ВІДМАЛЬОВКА КАДРУ
// ================================================================
function drawAnimFrame3D(fiFloat, hit) {
  if (!scene3D) ensure3DScene(G.currentActive3DContainer || 'field-anim-container');

  const bx = lerpVal(G.fullDf.ball.x, fiFloat);
  const by = lerpVal(G.fullDf.ball.y, fiFloat);
  const bz = lerpVal(G.fullDf.ball.z, fiFloat);
  ball3D.position.set(bx, bz + 100, by);

  if (!scene3D.userData.aiLine) {
    // Створюємо пунктирну неонову лінію (ініціалізація, якщо немає)
    const mat = new THREE.LineDashedMaterial({ color: 0x22c55e, dashSize: 150, gapSize: 100, linewidth: 2, transparent: true, opacity: 0.9 });
    const geo = new THREE.BufferGeometry();
    const line = new THREE.Line(geo, mat);
    scene3D.add(line);
    scene3D.userData.aiLine = line;

    const targetGeo = new THREE.RingGeometry(80, 120, 32);
    const targetMat = new THREE.MeshBasicMaterial({ color: 0x22c55e, transparent: true, opacity: 0.8, side: THREE.DoubleSide });
    const targetMesh = new THREE.Mesh(targetGeo, targetMat);
    targetMesh.rotation.x = -Math.PI / 2;
    scene3D.add(targetMesh);
    scene3D.userData.aiTarget = targetMesh;
    
    const aiLabelDiv = document.createElement('div');
    aiLabelDiv.id = 'ai-3d-label';
    aiLabelDiv.textContent = '💡 AI OPTIMAL';
    aiLabelDiv.className = 'ai-3d-floating-label';
    aiLabelDiv.style.position = 'absolute';
    aiLabelDiv.style.background = 'rgba(34,197,94,0.15)';
    aiLabelDiv.style.border = '1px solid rgba(34,197,94,0.5)';
    aiLabelDiv.style.color = '#4ade80';
    aiLabelDiv.style.padding = '4px 8px';
    aiLabelDiv.style.borderRadius = '4px';
    aiLabelDiv.style.fontFamily = "'Space Mono', monospace";
    aiLabelDiv.style.fontSize = '11px';
    aiLabelDiv.style.fontWeight = 'bold';
    aiLabelDiv.style.pointerEvents = 'none';
    aiLabelDiv.style.transform = 'translate(-50%, -50%)';
    aiLabelDiv.style.backdropFilter = 'blur(2px)';
    aiLabelDiv.style.display = 'none';
    
    const uiLayer = document.getElementById('anim-ui-layer');
    if (uiLayer) uiLayer.appendChild(aiLabelDiv);
    scene3D.userData.aiLabelDiv = aiLabelDiv;
  }

  const aiLine = scene3D.userData.aiLine;
  const aiTarget = scene3D.userData.aiTarget;
  const aiLabelDiv = scene3D.userData.aiLabelDiv;

  // Візуалізація AI Траєкторії (тільки якщо є активний hit)
  if (G.currentAIPredict && hit && Math.abs(fiFloat - hit.frame) < 150) {
    const traj = G.currentAIPredict;
    const pV_X = traj.pvx || 0;
    const pV_Y = traj.pvy || 0;
    const dist = Math.sqrt((traj.toX - traj.fromX)**2 + (traj.toY - traj.fromY)**2);
    const speedScale = Math.sqrt(pV_X**2 + pV_Y**2) || 1;
    
    const cpX = traj.fromX + (pV_X / speedScale) * (dist * 0.4);
    const cpY = traj.fromY + (pV_Y / speedScale) * (dist * 0.4);

    const curve = new THREE.QuadraticBezierCurve3(
      new THREE.Vector3(traj.fromX, 20, traj.fromY),
      new THREE.Vector3(cpX, 20, cpY),
      new THREE.Vector3(traj.toX, 20, traj.toY)
    );
    
    aiLine.geometry.setFromPoints(curve.getPoints(20));
    aiLine.computeLineDistances(); 
    aiLine.visible = true;

    aiTarget.position.set(traj.toX, 20, traj.toY);
    aiTarget.scale.setScalar(1 + Math.sin(fiFloat * 0.15) * 0.2); 
    aiTarget.visible = true;

    const pos2D = aiTarget.position.clone();
    pos2D.y += 120;
    pos2D.project(camera3D);
    if (pos2D.z < 1 && aiLabelDiv) {
      aiLabelDiv.style.left = `${(pos2D.x * .5 + .5) * renderer3D.domElement.clientWidth}px`;
      aiLabelDiv.style.top = `${(pos2D.y * -.5 + .5) * renderer3D.domElement.clientHeight}px`;
      aiLabelDiv.style.display = 'block';
    } else if (aiLabelDiv) {
      aiLabelDiv.style.display = 'none';
    }
  } else {
    if(aiLine) aiLine.visible = false;
    if(aiTarget) aiTarget.visible = false;
    if(aiLabelDiv) aiLabelDiv.style.display = 'none';
  }

  const W = renderer3D.domElement.clientWidth;
  const H = renderer3D.domElement.clientHeight;
  const nameplatesDiv = document.getElementById('ui-nameplates');

  let blueBoostsHtml = '';
  let orangeBoostsHtml = '';

  G.players.forEach(p => {
    const px = lerpVal(G.fullDf[p.name]?.x, fiFloat);
    const py = lerpVal(G.fullDf[p.name]?.y, fiFloat);
    const pz = lerpVal(G.fullDf[p.name]?.z, fiFloat);
    const vx = lerpVal(G.fullDf[p.name]?.vx, fiFloat);
    const vy = lerpVal(G.fullDf[p.name]?.vy, fiFloat);
    const vz = lerpVal(G.fullDf[p.name]?.vz, fiFloat);

    const qx = lerpVal(G.fullDf[p.name]?.qx, fiFloat);
    const qy = lerpVal(G.fullDf[p.name]?.qy, fiFloat);
    const qz = lerpVal(G.fullDf[p.name]?.qz, fiFloat);
    const qw = lerpVal(G.fullDf[p.name]?.qw, fiFloat);

    const mesh = getOrCreatePlayerMesh(p);
    mesh.position.set(px, pz + 10, py);
    
    if (!isNaN(qx) && (qx !== 0 || qy !== 0 || qz !== 0)) {
      mesh.quaternion.set(qx, qz, qy, -qw);
      mesh.rotateY(-Math.PI / 2);
    } else {
      if (Math.abs(vx) > 10 || Math.abs(vy) > 10 || Math.abs(vz) > 10) {
        mesh.lookAt(px + vx, pz + 10 + vz, py + vy);
      }
    }
    
    const f0 = Math.floor(fiFloat);
    const getB = (idx) => typeof getV === 'function' ? getV(G.fullDf[p.name]?.boost, idx) : 0;
    const bCurrent = getB(f0);
    const bPrev = getB(Math.max(0, f0 - 1));
    const isBoosting = bCurrent < bPrev && bCurrent > 0;

    if (mesh.userData.flame) {
      mesh.userData.flame.visible = isBoosting;
      if (isBoosting) mesh.userData.flame.scale.set(1, 0.8 + Math.random() * 0.6, 1);
    }

    if (mesh.userData.trailMeshes) {
      const trailLen = mesh.userData.trailMeshes.length;
      for (let i = 0; i < trailLen; i++) {
        const tMesh = mesh.userData.trailMeshes[i];
        const tPast = fiFloat - (i * 0.4); 
        if (tPast < 0) { tMesh.visible = false; continue; }

        const fPast = Math.floor(tPast);
        const bPastCurr = getB(fPast);
        const bPastPrev = getB(Math.max(0, fPast - 1));
        const wasBoosting = bPastCurr < bPastPrev && bPastCurr > 0;

        if (!wasBoosting) {
          tMesh.visible = false;
        } else {
          const ptX = lerpVal(G.fullDf[p.name]?.x, tPast);
          const ptY = lerpVal(G.fullDf[p.name]?.y, tPast);
          const ptZ = lerpVal(G.fullDf[p.name]?.z, tPast);
          const ptVx = lerpVal(G.fullDf[p.name]?.vx, tPast);
          const ptVy = lerpVal(G.fullDf[p.name]?.vy, tPast);
          const ptVz = lerpVal(G.fullDf[p.name]?.vz, tPast);

          const MathHypot = Math.sqrt(ptVx**2 + ptVy**2 + ptVz**2) || 1;
          const offset = 80;
          tMesh.position.set(ptX - (ptVx/MathHypot) * offset, ptZ + 15 - (ptVz/MathHypot) * offset, ptY - (ptVy/MathHypot) * offset);
          const scale = 1 - (i / trailLen);
          tMesh.scale.set(scale, scale, scale);
          tMesh.material.opacity = scale * 0.6;
          tMesh.visible = true;
        }
      }
    }

    const isActive = hit ? (p.name === hit.player) : false;
    
    if (mesh.userData.hitRing) {
      mesh.userData.hitRing.material.opacity = isActive ? 0.9 : 0;
    }
    if (mesh.userData.bodyMesh) {
      const baseColor = p.color === 'blue' ? 0x3b82f6 : 0xf97316;
      mesh.userData.bodyMesh.material.emissive = new THREE.Color(isActive ? 0xeab308 : baseColor);
      mesh.userData.bodyMesh.material.emissiveIntensity = isActive ? 0.4 : 0.15;
    }

    const pos = mesh.position.clone();
    pos.y += 200; 
    pos.project(camera3D);

    let np = playerNameplates[p.name];
    if (!np && nameplatesDiv) {
      const isBlue = p.color === 'blue';
      np = document.createElement('div');
      np.textContent = p.name;
      np.className = 'player-nameplate';
      np.style.position = 'absolute'; 
      np.style.transform = 'translate(-50%, -50%)';
      np.style.background = isBlue ? 'rgba(37, 99, 235, 0.85)' : 'rgba(234, 88, 12, 0.85)'; 
      np.style.border = `1px solid ${isBlue ? '#93c5fd' : '#fdba74'}`;
      np.style.color = '#ffffff'; 
      np.style.padding = '2px 5px';
      np.style.borderRadius = '4px';
      np.style.fontFamily = "'Space Mono', monospace"; 
      np.style.fontSize = '9px';
      np.style.fontWeight = 'bold';
      np.style.backdropFilter = 'blur(4px)';
      np.style.whiteSpace = 'nowrap';
      nameplatesDiv.appendChild(np);
      playerNameplates[p.name] = np;
    }

    if (np) {
      if (pos.z < 1) {
        np.style.left = `${(pos.x * .5 + .5) * W}px`;
        np.style.top = `${(pos.y * -.5 + .5) * H}px`;
        np.style.display = 'block';
      } else {
        np.style.display = 'none';
      }
    }

    const boostDisplay = Math.round(bCurrent);
    const bColor = p.color === 'blue' ? '#3b82f6' : '#f97316';
    const isBlue = p.color === 'blue';
    const boostRow = `
      <div style="background:rgba(10,14,22,0.6); backdrop-filter:blur(4px); padding: 4px 8px; border-radius: 6px; font-size: 12px; color: white; display: flex; gap: 8px; align-items: center; flex-direction: ${isBlue ? 'row' : 'row-reverse'}; border: 1px solid rgba(255,255,255,0.05); border-${isBlue ? 'left' : 'right'}: 3px solid ${bColor};">
        <span style="font-family: 'Barlow Condensed', sans-serif; font-size: 14px; min-width: 60px; text-align: ${isBlue ? 'left' : 'right'};">${p.name}</span>
        <div style="width: 50px; height: 8px; background: rgba(0,0,0,0.5); border-radius: 4px; overflow: hidden;">
          <div style="width: ${boostDisplay}%; height: 100%; background: ${bColor}; float: ${isBlue ? 'left' : 'right'};"></div>
        </div>
        <span style="width: 28px; font-weight: bold; font-family: 'Space Mono', monospace; color: ${boostDisplay < 20 ? '#ef4444' : '#f8fafc'}; text-align: ${isBlue ? 'right' : 'left'};">${boostDisplay}</span>
      </div>`;
    if (isBlue) blueBoostsHtml += boostRow; else orangeBoostsHtml += boostRow;
  });

  const bbLayer = document.getElementById('ui-blue-boosts');
  if (bbLayer) bbLayer.innerHTML = blueBoostsHtml;
  const obLayer = document.getElementById('ui-orange-boosts');
  if (obLayer) obLayer.innerHTML = orangeBoostsHtml;

  renderer3D.render(scene3D, camera3D);
}

// ================================================================
// FIELD ANIMATION & TIMELINE
// ================================================================

// Функція для ручного перемотування
function seekAnim(val) {
  const anim = G.animState;
  if (!G.currentMoment) return;
  
  // Якщо анімація зараз грає — ставимо на паузу, щоб не конфліктувати з мишкою
  if (anim.playing) {
    cancelAnimationFrame(anim.rafId);
    anim.playing = false;
    document.getElementById('anim-play-btn').textContent = '▶ Грати';
  }
  
  // Встановлюємо новий кадр
  anim.frame = parseFloat(val);
  
  // Оновлюємо текст часу
  const curTimeEl = document.getElementById('anim-time-current');
  if (curTimeEl) curTimeEl.textContent = formatGameTime(anim.frame);

  // Відмальовуємо 3D сцену на вибраному кадрі
  drawAnimFrame3D(anim.frame, G.currentMoment);
}

function startAnim(hit) {
  const anim = G.animState;
  if(anim.rafId) cancelAnimationFrame(anim.rafId);
  anim.targetFrame = hit.frame;
  anim.start = Math.max(0, hit.frame - 90);
  anim.end   = Math.min((G.fullDf.ball.x||[]).length-1, hit.frame + 90);
  anim.frame = anim.start;
  anim.playing = false;
  
  // Налаштовуємо повзунок під новий момент
  const slider = document.getElementById('anim-timeline');
  if (slider) {
    slider.min = anim.start;
    slider.max = anim.end;
    slider.value = anim.start;
  }
  
  // Оновлюємо текст стартового та кінцевого часу
  const curTimeEl = document.getElementById('anim-time-current');
  const endTimeEl = document.getElementById('anim-time-end');
  if (curTimeEl) curTimeEl.textContent = formatGameTime(anim.start);
  if (endTimeEl) endTimeEl.textContent = formatGameTime(anim.end);

  drawAnimFrame3D(anim.frame, hit); 
  document.getElementById('anim-play-btn').textContent = '▶ Грати';
}

// 3. Оновлений цикл (рухає повзунок під час відтворення)
function toggleAnim() {
  const anim = G.animState;
  if(anim.playing){
    cancelAnimationFrame(anim.rafId);
    anim.playing=false;
    document.getElementById('anim-play-btn').textContent='▶ Грати';
  } else {
    anim.playing=true;
    document.getElementById('anim-play-btn').textContent='⏸ Пауза';
    
    let lastTs = performance.now();

    function tick(ts){
      if (!G.currentMoment) return;
      
      const dt = ts - lastTs;
      lastTs = ts;

      anim.frame += dt * 0.03;

      if(anim.frame >= anim.end){ anim.frame = anim.start; }
      
      // Автоматично рухаємо повзунок та оновлюємо час
      const slider = document.getElementById('anim-timeline');
      if (slider) slider.value = anim.frame;
      
      const curTimeEl = document.getElementById('anim-time-current');
      if (curTimeEl) curTimeEl.textContent = formatGameTime(anim.frame);

      drawAnimFrame3D(anim.frame, G.currentMoment);

      if(anim.playing) anim.rafId = requestAnimationFrame(tick);
    }
    anim.rafId = requestAnimationFrame(tick);
  }
}

function showOptimalFrame() {
  if (!G.currentMoment) return;
  const anim = G.animState;
  
  if (anim.playing) {
    cancelAnimationFrame(anim.rafId);
    anim.playing = false;
    document.getElementById('anim-play-btn').textContent = '▶ Грати';
  }
  
  anim.frame = G.currentMoment.frame;
  
  // Синхронізуємо UI
  const slider = document.getElementById('anim-timeline');
  if (slider) slider.value = anim.frame;
  
  const curTimeEl = document.getElementById('anim-time-current');
  if (curTimeEl) curTimeEl.textContent = formatGameTime(anim.frame);

  drawAnimFrame3D(anim.frame, G.currentMoment);
}

// ================================================================
// DASHBOARD CHARTS & UI RENDERERS
// ================================================================
function renderAll() {
  renderScoreHero();
  renderMetrics();
  renderStatsTables();
  renderSpeedChart();
  renderHeatmap();
  renderBoostCharts();
  renderTiltChart();
  renderMomentChips();
  renderDetailedStats();
  drawRelativePos();
  renderPositioningBars();
  renderPlaystyleTab();
}

function renderScoreHero() {
  const blue = G.players.filter(p=>p.color==='blue');
  const orange = G.players.filter(p=>p.color==='orange');
  const bg = G.matchInfo.blueGoals;
  const og = G.matchInfo.orangeGoals;

  document.getElementById('blue-goals').textContent = bg;
  document.getElementById('orange-goals').textContent = og;
  document.getElementById('blue-players').textContent = blue.map(p=>p.name).join(' · ');
  document.getElementById('orange-players').textContent = orange.map(p=>p.name).join(' · ');

  // Час матчу
  const tot = Math.floor(G.matchInfo.totalSecs || 300);
  const m = Math.floor(tot / 60);
  const s = tot % 60;
  const timeEl = document.getElementById('m-time');
  if (timeEl) timeEl.textContent = `${m}:${s.toString().padStart(2,'0')}`;

  // Бейджі типу матчу + овертайм
  let modeHtml = '';
  if (G.matchInfo.isTournament) {
    modeHtml += '<span class="badge" style="background:rgba(234,179,8,.2);color:#eab308;border:1px solid rgba(234,179,8,.4)">TOURNAMENT</span> ';
  } else if (G.matchInfo.isRanked) {
    modeHtml += '<span class="badge" style="background:rgba(59,130,246,.2);color:#60a5fa;border:1px solid rgba(59,130,246,.4)">RANKED</span> ';
  } else if (G.matchInfo.isPrivate) {
    modeHtml += '<span class="badge" style="background:rgba(168,85,247,.2);color:#c084fc;border:1px solid rgba(168,85,247,.4)">PRIVATE</span> ';
  } else {
    modeHtml += '<span class="badge" style="background:rgba(34,197,94,.2);color:#4ade80;border:1px solid rgba(34,197,94,.4)">CASUAL</span> ';
  }
  modeHtml += `<span class="badge" style="background:rgba(255,255,255,.05);color:#94a3b8;border:1px solid var(--border);margin-left:4px">${G.matchInfo.gameMode}</span>`;

  if (G.matchInfo.isOvertime) {
    const otM = Math.floor(G.matchInfo.otSecs / 60);
    const otS = G.matchInfo.otSecs % 60;
    modeHtml += ` <span class="badge" style="background:rgba(249,115,22,.2);color:#fb923c;border:1px solid rgba(249,115,22,.4);margin-left:4px">OT +${otM}:${otS.toString().padStart(2,'0')}</span>`;
  }

  document.getElementById('match-ot-badge').innerHTML = modeHtml;

  // Назва файлу в topbar
  const replayNameEl = document.getElementById('replay-name');
  if (replayNameEl) {
    replayNameEl.innerHTML = `<span style="color:var(--muted);font-size:10px;margin-right:4px">FILE:</span>${G.replayFileName || '—'}`;
  }

  // Виграш — підсвітка блоку
  const hero = document.getElementById('score-hero');
  const tBlocks = hero.querySelectorAll('.team-block');
  tBlocks.forEach(b => {
    b.style.border = '2px solid transparent';
    b.style.borderRadius = '12px';
    b.style.padding = '12px 0';
    b.style.background = 'transparent';
  });
  if (bg > og) {
    tBlocks[0].style.border = '2px solid var(--blue)';
    tBlocks[0].style.background = 'rgba(59,130,246,0.05)';
  } else if (og > bg) {
    tBlocks[1].style.border = '2px solid var(--orange)';
    tBlocks[1].style.background = 'rgba(249,115,22,0.05)';
  }
}
function renderMetrics() {
  document.getElementById('m-touches').textContent = G.hits.length;

  let fastestSpeed=0, fastestName='—';
  G.players.forEach(p=>{
    const arr=G.fullDf[p.name]?.speed||[];
    const avg = arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0;
    if(avg>fastestSpeed){ fastestSpeed=avg; fastestName=p.name; }
  });
  document.getElementById('m-fastest').textContent = fastestSpeed.toFixed(0);
  document.getElementById('m-fastest-name').textContent = fastestName;

  const totalXg = G.players.reduce((a,p)=>a+p.xg,0);
  document.getElementById('m-xg').textContent = totalXg.toFixed(2);
}

function calcRating(p) {
  let r = 6;
  r += p.goals*0.8 + (p.goals - p.xg)*0.5;
  r += (p.assists - 0.6)*0.5 + (p.saves - 1.5)*0.3 + (p.shots - 3)*0.1 + (p.demos - 1.2)*0.15;
  if(p.shots>3 && p.goals===0) r-=0.5;
  return Math.max(1,Math.min(10,r));
}

function renderStatsTables() {
  const grid = document.getElementById('stats-grid');
  grid.innerHTML = '';
  ['blue','orange'].forEach(team => {
    const label = team==='blue'?'СИНІ':'ПОМАРАНЧЕВІ';
    const cls = team;
    const tPlayers = G.players.filter(p=>p.color===team).sort((a,b)=>calcRating(b)-calcRating(a));
    const card = document.createElement('div');
    card.className = 'stats-card';
    const maxR = tPlayers.length ? Math.max(...tPlayers.map(calcRating)) : 0;
    card.innerHTML = `
      <div class="stats-card-header ${cls}">${label}</div>
      <table>
        <thead>
          <tr>
            <th>Гравець</th>
            <th>Rating</th>
            <th>Score</th>
            <th>xG</th>
            <th>G</th><th>A</th><th>S</th><th>Sh</th>
          </tr>
        </thead>
        <tbody>
          ${tPlayers.map(p=>{
            const r=calcRating(p);
            return `<tr>
              <td class="td-name">${p.name}</td>
              <td class="td-rating ${r===maxR?'top':''}">${r.toFixed(2)}</td>
              <td>${p.score}</td>
              <td class="td-xg">${p.xg.toFixed(2)}</td>
              <td>${p.goals}</td><td>${p.assists}</td><td>${p.saves}</td><td>${p.shots}</td>
            </tr>`;
          }).join('')}
        </tbody>
      </table>`;
    grid.appendChild(card);
  });
}

function renderDetailedStats() {
  const grid = document.getElementById('detailed-stats-grid');
  grid.innerHTML = '';
  ['blue', 'orange'].forEach(team => {
    const tPlayers = G.players.filter(p => p.color === team).sort((a, b) => {
      return team === 'blue' ? b.relY - a.relY : a.relY - b.relY;
    });
    if(!tPlayers.length) return;
    
    const label = team === 'blue' ? 'СИНІ' : 'ПОМАРАНЧЕВІ';
    
    let html = `
    <div class="stats-card">
      <div class="stats-card-header ${team}">${label}</div>
      <table>
        <thead>
          <tr>
            <th>Метрика</th>
            ${tPlayers.map((p, idx) => `<th>[${idx + 1}] ${p.name}</th>`).join('')}
          </tr>
        </thead>
        <tbody>
          <tr><td class="td-name">Найближче до м'яча</td>${tPlayers.map(p => `<td>${p.pctClosest}</td>`).join('')}</tr>
          <tr><td class="td-name">Найдалі від м'яча</td>${tPlayers.map(p => `<td>${p.pctFurthest}</td>`).join('')}</tr>
          <tr><td class="td-name">Сер. дист. до м'яча</td>${tPlayers.map(p => `<td>${p.avgDist}</td>`).join('')}</tr>
          <tr><td class="td-name">Найбільш попереду</td>${tPlayers.map(p => `<td>${p.pctForward}</td>`).join('')}</tr>
          <tr><td class="td-name">Найбільш позаду (Last Man)</td>${tPlayers.map(p => `<td>${p.pctBack}</td>`).join('')}</tr>
          <tr><td class="td-name">Kickoff Wins / Total</td>${tPlayers.map(p => `<td>${p.kickoffWins} / ${p.kickoffTouches}</td>`).join('')}</tr>
        </tbody>
      </table>
    </div>`;
    grid.innerHTML += html;
  });
}

function drawRelativePos() {
  const canvas = document.getElementById('pos-canvas');
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height; 
  ctx.clearRect(0,0,W,H);
  
  const legendContainer = document.getElementById('pos-map-legend');
  if (legendContainer) legendContainer.innerHTML = '';
  let legendHtml = '';

  ['blue', 'orange'].forEach(team => {
    const members = G.players.filter(p => p.color === team).sort((a, b) => team === 'blue' ? b.relY - a.relY : a.relY - b.relY);
    if(!members.length) return;
    
    const teamTitle = team === 'blue' ? 'Синя команда' : 'Помаранчева команда';
    const bgStyle = team === 'blue' ? 'var(--blue-dim)' : 'var(--orange-dim)';
    const borderStyle = team === 'blue' ? 'var(--blue)' : 'var(--orange)';

    legendHtml += `<div style="display: flex; flex-direction: column; gap: 6px;">
        <div style="font-size: 11px; font-weight: 700; color: var(--muted); letter-spacing: 1px; text-transform: uppercase;">${teamTitle}</div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
          ${members.map((p, idx) => `
            <div style="background: ${bgStyle}; border: 1px solid ${borderStyle}; padding: 4px 10px; border-radius: 6px; display: flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 600; color: #fff;">
              <span style="background: rgba(255,255,255,0.18); padding: 1px 6px; border-radius: 4px; font-family: 'Space Mono', monospace; font-size: 11px; font-weight: 700;">${idx + 1}</span>
              <span>${p.name}</span>
            </div>
          `).join('')}
        </div>
      </div>`;

    const SCALE = W / 6500; 
    const anchorX = W / 2;
    const anchorY = team === 'blue' ? H * 0.75 : H * 0.25; // Сині знизу, Оранж зверху

    const SPREAD_X = 2.5;
    const SPREAD_Y = 4.5;

    members.forEach((p, i) => {
       let dx = p.relX * SCALE * SPREAD_X;
       let dy = -(p.relY * SCALE * SPREAD_Y); 

       // Обрізаємо максимальне відхилення, щоб вони точно не вилазили за межі поля
       dx = Math.max(-100, Math.min(100, dx));
       dy = Math.max(-120, Math.min(120, dy));
       
       const cx = anchorX + dx;
       const cy = anchorY + dy;

       ctx.beginPath();
       ctx.arc(cx, cy, 14, 0, Math.PI*2);
       ctx.fillStyle = p.color === 'blue' ? '#3b82f6' : '#f97316';
       ctx.fill();
       ctx.lineWidth = 2;
       ctx.strokeStyle = 'white';
       ctx.stroke();

       ctx.fillStyle = 'white';
       ctx.font = 'bold 12px Space Mono, monospace';
       ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
       ctx.fillText((i + 1).toString(), cx, cy);

       const lw = ctx.measureText(p.name).width + 10;
       ctx.fillStyle = 'rgba(0,0,0,0.6)';
       ctx.fillRect(cx - lw/2, cy + 18, lw, 14);

       ctx.fillStyle = 'white';
       ctx.font = '10px Barlow Condensed, sans-serif';
       ctx.fillText(p.name, cx, cy + 25);
    });
  });
  if (legendContainer) legendContainer.innerHTML = legendHtml;
}

function renderSpeedChart() {
  const avgSpeeds = G.players.map(p => {
    const arr = G.fullDf[p.name]?.speed||[];
    return arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0;
  });
  const sorted = [...G.players].map((p,i)=>({p,s:avgSpeeds[i]})).sort((a,b)=>b.s-a.s);

  destroyChart('speed');
  G.chartInstances['speed'] = new Chart(document.getElementById('speed-chart'), {
    type: 'bar',
    data: {
      labels: sorted.map(x=>x.p.name),
      datasets: [{
        data: sorted.map(x=>x.s),
        backgroundColor: sorted.map(x=>x.p.color==='blue'?'rgba(59,130,246,.7)':'rgba(249,115,22,.7)'),
        borderColor: sorted.map(x=>x.p.color==='blue'?'#3b82f6':'#f97316'),
        borderWidth: 1.5,
        borderRadius: 4,
      }]
    },
    options: {
      responsive:true, 
      maintainAspectRatio:false,
      plugins:{ legend:{display:false} },
      scales:{
        x:{grid:{display:false},ticks:{color:'#94a3b8',font:{family:'Space Mono',size:11}}},
        y:{grid:{color:'rgba(255,255,255,.05)'},ticks:{color:'#64748b',font:{size:11}},
           title:{display:true,text:'км/год',color:'#64748b'}}
      }
    }
  });
}

// ================================================================
// HEATMAP
// ================================================================

function drawField(ctx, w, h) {
  if (G.fieldImg && G.fieldImg.complete && G.fieldImg.naturalWidth !== 0) {
    ctx.drawImage(G.fieldImg, 0, 0, w, h);
  } else {
    // Резервний варіант фону, якщо SVG-зображення поля не завантажилось
    ctx.fillStyle = '#0a0e16';
    ctx.fillRect(0, 0, w, h);
  }
}

function renderHeatmap() {
  const select = document.getElementById('heatmap-select');
  if(select.options.length === 3) {
    G.players.forEach(p=>{
      const o=document.createElement('option');
      o.value=p.name; o.textContent=p.name;
      select.appendChild(o);
    });
  }

  select.onchange = drawHeatmap;
  document.getElementById('heatmap-res').oninput = function(){
    document.getElementById('res-val').textContent = this.value;
    drawHeatmap();
  };
  drawHeatmap();
  drawTouchMap();
}

function getHeatmapData() {
  const val = document.getElementById('heatmap-select').value;
  let xs=[], ys=[];
  if(val==='all') {
    G.players.forEach(p=>{ xs=xs.concat(G.paths[p.name].x); ys=ys.concat(G.paths[p.name].y); });
  } else if(val==='blue'||val==='orange') {
    G.players.filter(p=>p.color===val).forEach(p=>{ xs=xs.concat(G.paths[p.name].x); ys=ys.concat(G.paths[p.name].y); });
  } else {
    const p=G.paths[val]||{x:[],y:[]};
    xs=p.x; ys=p.y;
  }
  return {xs,ys};
}

function drawHeatmap() {
  const canvas = document.getElementById('heatmap-canvas');
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  const W=canvas.width, H=canvas.height; // W=480, H=340
  ctx.clearRect(0,0,W,H);
  drawField(ctx,W,H);

  const {xs,ys} = getHeatmapData();
  if(!xs.length) return;

  const res = parseInt(document.getElementById('heatmap-res').value);
  const cellW = W/res, cellH = H/res;
  const grid = new Float32Array(res*res);
  let maxVal=0;

  xs.forEach((x,i)=>{
    const y=ys[i];
    if (x == null || y == null || (Math.abs(x) < 1 && Math.abs(y) < 1)) return;
    const gx = Math.floor((y + 5120) / (10240 / res));
    const gy = Math.floor((x + 4096) / (8192 / res));
    const cx=Math.max(0,Math.min(res-1,gx));
    const cy=Math.max(0,Math.min(res-1,gy));
    grid[cy*res+cx]++;
    if(grid[cy*res+cx]>maxVal) maxVal=grid[cy*res+cx];
  });
  if(maxVal===0) return;
  const logMax=Math.log(maxVal+1);

  for(let cy=0;cy<res;cy++) for(let cx=0;cx<res;cx++){
    const v=grid[cy*res+cx];
    if(v===0) continue;
    const t=Math.log(v+1)/logMax;
    ctx.globalAlpha = t*0.85;
    ctx.fillStyle = heatColor(t);
    ctx.fillRect(cx*cellW, cy*cellH, cellW+1, cellH+1);
  }
  ctx.globalAlpha=1;
  const total = ys.length;
  const bluePct   = ys.filter(y => y < -1706).length / total * 100;
  const midPct    = ys.filter(y => y >= -1706 && y <= 1706).length / total * 100;
  const orangePct = ys.filter(y => y > 1706).length / total * 100;
  // Лінії зон тепер вертикальні (бо Y гри = горизонталь canvas)
  const toCanvasX = y => (y + 5120) / 10240 * W;
  ctx.strokeStyle = 'rgba(255,255,255,.4)'; 
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4,4]);
  ctx.beginPath();
  ctx.moveTo(toCanvasX(-1706), 0); ctx.lineTo(toCanvasX(-1706), H); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(toCanvasX(1706), 0);  ctx.lineTo(toCanvasX(1706), H);  ctx.stroke();
  ctx.setLineDash([]);

  const val = document.getElementById('heatmap-select').value;
  let teamContext = 'all';
  if (val === 'blue' || val === 'orange') {
    teamContext = val;
  } else if (val !== 'all') {
    const p = G.players.find(x => x.name === val);
    if (p) teamContext = p.color;
  }

  let blueLabel = "Синя пол.";
  let orangeLabel = "Оранж. пол.";
  if (teamContext === 'blue') {
    blueLabel = "Захист";
    orangeLabel = "Атака";
  } else if (teamContext === 'orange') {
    blueLabel = "Атака";
    orangeLabel = "Захист";
  }

  document.getElementById('zone-left').textContent  = `${blueLabel} ${bluePct.toFixed(0)}%`;
  document.getElementById('zone-mid').textContent   = `Центр ${midPct.toFixed(0)}%`;
  document.getElementById('zone-right').textContent = `${orangeLabel} ${orangePct.toFixed(0)}%`;
}

function drawTouchMap() {
  const canvas = document.getElementById('touch-canvas');
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  const W=canvas.width, H=canvas.height; // W=480, H=340
  ctx.clearRect(0,0,W,H);
  drawField(ctx,W,H);
  G.hits.forEach(h=>{
    if (Math.abs(h.ballX) < 1 && Math.abs(h.ballY) < 1) return;
    const cx=(h.ballY+5120)/(10240)*W;
    const cy=(h.ballX+4096)/(8192)*H;
    ctx.beginPath();
    ctx.arc(cx,cy,5,0,Math.PI*2);
    ctx.fillStyle=h.color==='blue'?'rgba(59,130,246,.7)':'rgba(249,115,22,.7)';
    ctx.fill();
    ctx.strokeStyle=h.color==='blue'?'#3b82f6':'#f97316';
    ctx.lineWidth=1;
    ctx.stroke();
  });
}

function heatColor(t) {
  if(t<0.25) return `rgba(0,0,255,${t*4})`;
  if(t<0.5) return `rgba(0,${Math.floor((t-.25)*4*255)},${255-Math.floor((t-.25)*4*255)},1)`;
  if(t<0.75) return `rgba(${Math.floor((t-.5)*4*255)},255,0,1)`;
  return `rgba(255,${255-Math.floor((t-.75)*4*255)},0,1)`;
}

// ================================================================
// BOOST CHARTS
// ================================================================
function renderBoostCharts() {
  const avgs = G.players.map(p=>{
    const arr=G.boostData[p.name]||[];
    return arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0;
  });
  const zeros = G.players.map(p=>{
    const arr=G.boostData[p.name]||[];
    return arr.length ? arr.filter(x=>x===0).length/arr.length*100 : 0;
  });
  const labels = G.players.map(p=>p.name);
  const bgs = G.players.map(p=>p.color==='blue'?'rgba(59,130,246,.7)':'rgba(249,115,22,.7)');

  destroyChart('boost-avg');
  G.chartInstances['boost-avg'] = new Chart(document.getElementById('boost-avg-chart'), {
    type:'bar',
    data:{ labels, datasets:[{ data:avgs, backgroundColor:bgs, borderRadius:4 }] },
    options:{
      indexAxis:'y', responsive:true, maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{
        x:{max:100,grid:{color:'rgba(255,255,255,.05)'},ticks:{color:'#64748b'}},
        y:{grid:{display:false},ticks:{color:'#94a3b8',font:{size:12}}}
      }
    }
  });
  destroyChart('boost-zero');
  G.chartInstances['boost-zero'] = new Chart(document.getElementById('boost-zero-chart'), {
    type:'bar',
    data:{ labels, datasets:[{ data:zeros, backgroundColor:'rgba(239,68,68,.6)', borderRadius:4 }] },
    options:{
      indexAxis:'y', responsive:true, maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{
        x:{grid:{color:'rgba(255,255,255,.05)'},ticks:{color:'#64748b',callback:v=>v+'%'}},
        y:{grid:{display:false},ticks:{color:'#94a3b8',font:{size:12}}}
      }
    }
  });
}

// ================================================================
// FIELD TILT
// ================================================================
function renderTiltChart() {
  const raw = G.ballYTimeline;
  if(!raw.length) return;
  const WINDOW = 150, STEP = 15;
  const smoothed = [], frameIndices = [];
  
  for (let i = WINDOW; i < raw.length; i += STEP) {
    const slice = raw.slice(i - WINDOW, i);
    smoothed.push(slice.reduce((a,b) => a+b, 0) / slice.length);
    frameIndices.push(i);
  }

  const blues = smoothed.map(v => v > 0 ? (v / 5120) * 100 : 0);
  const oranges = smoothed.map(v => v < 0 ? (Math.abs(v) / 5120) * 100 : 0);

  const legendEl = document.querySelector('.tilt-legend');
  if (legendEl) {
     legendEl.innerHTML = `
       <span><span class="tilt-dot" style="background:rgba(59,130,246,.7)"></span>Сині тиснуть (Атака)</span>
       <span><span class="tilt-dot" style="background:rgba(249,115,22,.7)"></span>Помаранчеві тиснуть (Атака)</span>
     `;
  }

  const eventData = {
    shots: { data: new Array(frameIndices.length).fill(null), meta: new Array(frameIndices.length).fill(null) },
    clears: { data: new Array(frameIndices.length).fill(null), meta: new Array(frameIndices.length).fill(null) },
    saves: { data: new Array(frameIndices.length).fill(null), meta: new Array(frameIndices.length).fill(null) },
    passes: { data: new Array(frameIndices.length).fill(null), meta: new Array(frameIndices.length).fill(null) },
    demos: { data: new Array(frameIndices.length).fill(null), meta: new Array(frameIndices.length).fill(null) }
  };

  function findFreeIndex(targetArr, startIdx) {
    let offset = 0;
    while (offset < 4) { 
      if (startIdx + offset < targetArr.length && targetArr[startIdx + offset] === null) return startIdx + offset;
      if (startIdx - offset >= 0 && targetArr[startIdx - offset] === null) return startIdx - offset;
      offset++;
    }
    return startIdx; 
  }

  G.hits.forEach(h => {
    const timeIdx = frameIndices.findIndex(fIdx => Math.abs(fIdx - h.frame) <= (STEP * 1.5));
    if (timeIdx !== -1) {
      const yValue = (smoothed[timeIdx] / 5120) * 100;
      let finalIdx;
      switch(h.action) {
        case 3: finalIdx = findFreeIndex(eventData.shots.data, timeIdx); eventData.shots.data[finalIdx] = yValue; eventData.shots.meta[finalIdx] = h; break;
        case 4: finalIdx = findFreeIndex(eventData.clears.data, timeIdx); eventData.clears.data[finalIdx] = yValue; eventData.clears.meta[finalIdx] = h; break;
        case 12: finalIdx = findFreeIndex(eventData.saves.data, timeIdx); eventData.saves.data[finalIdx] = yValue; eventData.saves.meta[finalIdx] = h; break;
        case 1: finalIdx = findFreeIndex(eventData.passes.data, timeIdx); eventData.passes.data[finalIdx] = yValue; eventData.passes.meta[finalIdx] = h; break;
        case 9: finalIdx = findFreeIndex(eventData.demos.data, timeIdx); eventData.demos.data[finalIdx] = yValue; eventData.demos.meta[finalIdx] = h; break;
      }
    }
  });

  destroyChart('tilt');
  G.chartInstances['tilt'] = new Chart(document.getElementById('tilt-chart'), {
    type:'line',
    data:{
      labels: frameIndices,
      datasets:[
        { label:'Сині тиснуть', data:blues, fill:true, backgroundColor:'rgba(59,130,246,.3)', borderColor:'rgba(59,130,246,.8)', borderWidth:1.5, pointRadius:0, tension:.4, order: 3 },
        { label:'Помаранчеві тиснуть', data:oranges.map(v=>-v), fill:true, backgroundColor:'rgba(249,115,22,.3)', borderColor:'rgba(249,115,22,.8)', borderWidth:1.5, pointRadius:0, tension:.4, order: 3 },
        
        { label: '⚽ Удар', data: eventData.shots.data, hitMeta: eventData.shots.meta, showLine: false, pointStyle: 'star', pointRadius: 10, backgroundColor: '#eab308', borderColor: '#ffffff', borderWidth: 1, order: 1 },
        { label: '🛡️ Клір', data: eventData.clears.data, hitMeta: eventData.clears.meta, showLine: false, pointStyle: 'rectRounded', pointRadius: 6, backgroundColor: '#3b82f6', borderColor: '#ffffff', borderWidth: 1, order: 2 },
        { label: '🧤 Сейв', data: eventData.saves.data, hitMeta: eventData.saves.meta, showLine: false, pointStyle: 'triangle', pointRadius: 8, backgroundColor: '#22c55e', borderColor: '#ffffff', borderWidth: 1, order: 1 },
        { label: '🎯 Пас', data: eventData.passes.data, hitMeta: eventData.passes.meta, showLine: false, pointStyle: 'circle', pointRadius: 5, backgroundColor: '#a855f7', borderColor: '#ffffff', borderWidth: 1, order: 2 },
        { label: '💥 Демо', data: eventData.demos.data, hitMeta: eventData.demos.meta, showLine: false, pointStyle: 'crossRot', pointRadius: 7, backgroundColor: '#ef4444', borderColor: '#ffffff', borderWidth: 2, order: 2 }
      ]
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      plugins:{
        legend:{display:false},
        tooltip: {
          callbacks: {
            title: function(ctx) { return `Час: ${formatGameTime(frameIndices[ctx[0].dataIndex])}`; },
            label: function(ctx) {
              const ds = ctx.chart.data.datasets[ctx.datasetIndex];
              if (ctx.datasetIndex >= 2 && ds.hitMeta && ds.hitMeta[ctx.dataIndex]) {
                 const hit = ds.hitMeta[ctx.dataIndex];
                 return `${ds.label.split(' ')[0]} ${hit.actionName} — ${hit.player}`;
              }
              const teamName = ctx.datasetIndex === 0 ? 'Сині' : 'Помаранчеві';
              return `Тиск (${teamName}): ${Math.abs(Math.round(ctx.raw))}%`;
            }
          }
        }
      },
      scales:{
        x: {
          type: 'category',
          grid: { display: false },
          ticks: {
            color: '#64748b', 
            maxTicksLimit: 12,
            callback: function(val) {
              // Надійний діставач значення: бере оригінальний кадр з масиву labels
              const frame = this.getLabelForValue(val);
              return frame !== undefined ? formatGameTime(frame) : '';
            }
          }
        },
        y:{ grid:{color:'rgba(255,255,255,.05)'}, ticks:{color:'#64748b',callback:v=>Math.abs(v).toFixed(0)+'%'}, min:-100, max:100 }
      }
    }
  });
}

function renderPositioningBars() {
  const container = document.getElementById('positioning-bars');
  if (!container) return;
  container.innerHTML = '';
  G.players.forEach(p => {
    const NF = G.fullDf[p.name]?.y?.length || 0;
    if (!NF) return;

    let defFrames = 0, midFrames = 0, offFrames = 0;

    for (let f = 0; f < NF; f++) {
      const py = G.fullDf[p.name].y[f];
      if (py == null) continue;

      let fieldY = p.color === 'blue' ? py : -py;

      if (fieldY < -1706)        defFrames++; 
      else if (fieldY <= 1706)   midFrames++; 
      else                       offFrames++; 
    }

    const total = Math.max(1, defFrames + midFrames + offFrames);
    const defPct = (defFrames / total * 100).toFixed(1);
    const midPct = (midFrames / total * 100).toFixed(1);
    const offPct = (offFrames / total * 100).toFixed(1);

    const nameColor  = p.color === 'blue' ? 'var(--blue)' : 'var(--orange)';

    container.innerHTML += `
      <div class="pos-bar-wrap">
        <div class="pos-bar-name" style="color:${nameColor}">${p.name}</div>
        <div class="pos-bar-track">
          <div class="pos-bar-seg def" style="width:${defPct}%">${defPct > 5 ? defPct + '%' : ''}</div>
          <div class="pos-bar-seg mid" style="width:${midPct}%">${midPct > 5 ? midPct + '%' : ''}</div>
          <div class="pos-bar-seg off" style="width:${offPct}%">${offPct > 5 ? offPct + '%' : ''}</div>
        </div>
        <div class="pos-bar-labels">
          <span>DEF</span><span>MID</span><span>OFF</span>
        </div>
      </div>
    `;
  });
}

function calcPlaystyleProfile(playerName) {
  const p = G.players.find(x => x.name === playerName);
  if (!p) return { aggression: 50, defense: 50, mechanics: 50, rotation: 50, boostMgmt: 50, teamplay: 50 };

  const matchMins = Math.max(1, (G.matchInfo?.totalSecs || 300) / 60);
  const hits = G.hits.filter(h => h.player === playerName);
  
  const shots = p.shots || 0;
  const goals = p.goals || 0;
  const saves = p.saves || 0;
  const assists = p.assists || 0;

  // Рахуємо кліри та паси з хітів для більшої точності
  const clears = hits.filter(h => h.action === 4).length;
  const passes = hits.filter(h => h.action === 1).length;
  
  // Відсоток перебування в зонах (0-100)
  const fwdPct = parseFloat(p.pctForward) || 0;
  const backPct = parseFloat(p.pctBack) || 0;
  
  // АГРЕСІЯ: Балансуємо множники. 1 удар = +8, 1 гол = +12.
  let agg = (shots * 8) + (goals * 12) + (fwdPct * 0.8);
  
  // ЗАХИСТ: 1 сейв = +12, 1 клір = +4.
  let def = (saves * 12) + (clears * 4) + (backPct * 0.8);
  
  // МЕХАНІКИ: Фікс "всі по 100". Тепер рахується тільки з ейріалів та небезпеки ударів (xG). База = 20.
  const aerials = hits.filter(h => {
    const bz = Number(typeof getV === 'function' ? getV(G.fullDf.ball?.z, h.frame) : (G.fullDf.ball?.z[h.frame] || 0));
    return bz > 300; 
  }).length;
  let mech = 20 + ((aerials / matchMins) * 8) + (p.xg * 25);
  
  // РОТАЦІЯ: Ідеальна ротація - це 33% часу в кожній зоні. 
  // Штрафуємо за відхилення від ідеалу.
  const devFwd = Math.abs(fwdPct - 33);
  const devBack = Math.abs(backPct - 33);
  let rot = 100 - ((devFwd + devBack) * 1.5); 
  
  // БУСТ МЕНЕДЖМЕНТ
  const boostData = G.boostData[playerName] || [];
  const avgBoost = boostData.length ? boostData.reduce((a,b)=>a+b,0)/boostData.length : 33;
  const zeroPct = boostData.length ? (boostData.filter(x=>x===0).length / boostData.length) * 100 : 50;
  let bst = (avgBoost * 1.6) - (zeroPct * 1.2);
  
  // КОМАНДНА ГРА: Базові 15 балів + потужний буст за асисти та невеликий за звичайні паси
  let tm = 15 + (assists * 25) + (passes * 3);

  // Жорсткий ліміт значень від 10 до 100
  const clamp = (v) => Math.min(100, Math.max(10, Math.round(v))); 

  return {
    aggression: clamp(agg),
    defense: clamp(def),
    mechanics: clamp(mech),
    rotation: clamp(rot),
    boostMgmt: clamp(bst),
    teamplay: clamp(tm)
  };
}

function getPlaystyleLabel(profile) {
  const { aggression, defense, mechanics, rotation, boostMgmt, teamplay } = profile;

  // Радикальні стилі
  if (aggression > 85 && rotation < 55) return 'М\'ячогон';
  if (defense > 75 && rotation > 60 && aggression < 60) return 'Якір (Third Man)';
  
  // Технічні та командні
  if (mechanics > 75 && aggression > 75) return 'Механічний Форвард';
  if (teamplay >= 70) return 'Плеймейкер';
  if (boostMgmt > 75 && mechanics > 65) return 'Ейріал-загроза';
  
  // Дисбаланс
  if (boostMgmt < 35 && aggression > 65) return 'Голодний Форвард';
  if (defense < 40 && teamplay < 45) return 'Соліст';
  
  // Збалансовані
  if (aggression >= 60 && defense >= 60 && rotation >= 60) return 'MVP Ротації';
  
  return 'Універсал';
}

function renderPlaystyleTab() {
  const container = document.getElementById('tab-playstyle');
  container.innerHTML = '';

  // === 1. БЛОК ПОРІВНЯННЯ ГРАВЦІВ ===
  const playersOptions = G.players.map(p => `<option value="${p.name}">${p.name}</option>`).join('');
  
  const compareSection = document.createElement('div');
  compareSection.className = 'chart-card';
  compareSection.style.marginBottom = '24px';
  compareSection.innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 16px;">
      <div class="section-title" style="margin:0;">📊 Порівняння гравців</div>
      <div style="display:flex; gap: 12px; align-items:center;">
        <select id="cmp-p1" style="background:var(--bg2); border:1px solid var(--border); color:var(--text); padding:6px 12px; border-radius:6px; outline:none; font-family:'Space Mono', monospace; font-size: 13px;">
          ${playersOptions}
        </select>
        <span style="color:var(--muted); font-weight:bold;">VS</span>
        <select id="cmp-p2" style="background:var(--bg2); border:1px solid var(--border); color:var(--text); padding:6px 12px; border-radius:6px; outline:none; font-family:'Space Mono', monospace; font-size: 13px;">
          ${playersOptions}
        </select>
      </div>
    </div>
    <div style="display:flex; justify-content:center; align-items:center; width:100%; height:320px;">
       <canvas id="radar-compare"></canvas>
    </div>
  `;
  container.appendChild(compareSection);

  // === 2. СІТКА ДЛЯ КАРТОК ГРАВЦІВ (По два в ряд) ===
  const gridContainer = document.createElement('div');
  gridContainer.style.display = 'grid';
  gridContainer.style.gridTemplateColumns = 'repeat(auto-fit, minmax(400px, 1fr))';
  gridContainer.style.gap = '20px';
  container.appendChild(gridContainer);

  G.players.forEach(p => {
    const profile = calcPlaystyleProfile(p.name);
    const isBlue = p.color === 'blue';
    const rgb = isBlue ? '59,130,246' : '249,115,22';
    const hexColor = isBlue ? '#3b82f6' : '#f97316';
    const label = getPlaystyleLabel(profile);

    const card = document.createElement('div');
    card.className = 'chart-card';
    card.style.margin = '0'; // Скидаємо відступ, бо тепер працює gap у гріді
    
    card.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <div class="chart-card-title" style="color: ${hexColor}; font-size: 18px; margin: 0;">${p.name}</div>
        <span style="background: rgba(${rgb}, 0.15); border: 1px solid rgba(${rgb}, 0.4); color: #fff; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">
          ${label}
        </span>
      </div>
      
      <div style="display: flex; gap: 16px; align-items: center;">
        <div style="width: 170px; height: 170px; flex-shrink: 0;">
          <canvas id="radar-${p.name.replace(/\s/g,'_')}"></canvas>
        </div>
        
        <div style="display: flex; flex-direction: column; gap: 10px; flex: 1;">
          ${Object.entries({
            'Агр.': profile.aggression,
            'Зах.': profile.defense,
            'Мех.': profile.mechanics,
            'Рот.': profile.rotation,
            'Буст': profile.boostMgmt,
            'Ком.': profile.teamplay,
          }).map(([lbl, val]) => `
            <div style="display: flex; align-items: center; gap: 8px;">
              <span style="width: 35px; font-size: 12px; font-weight: 600; color: var(--muted);">${lbl}</span>
              <div style="flex: 1; height: 6px; background: var(--bg3); border-radius: 3px; overflow: hidden;">
                <div style="width: ${val}%; height: 100%; background: ${hexColor}; border-radius: 3px;"></div>
              </div>
              <span style="width: 28px; text-align: right; font-size: 13px; font-weight: bold; font-family: 'Space Mono', monospace; color: #fff;">${val}</span>
            </div>
          `).join('')}
        </div>
      </div>
    `;
    gridContainer.appendChild(card);

    // Малюємо маленький радар для кожного гравця
    requestAnimationFrame(() => {
      const canvas = document.getElementById(`radar-${p.name.replace(/\s/g,'_')}`);
      if (!canvas) return;
      new Chart(canvas, {
        type: 'radar',
        data: {
          labels: ['Агресія', 'Захист', 'Механіки', 'Ротація', 'Буст', 'Команда'],
          datasets: [{
            data: Object.values(profile),
            backgroundColor: `rgba(${rgb}, 0.2)`,
            borderColor: hexColor,
            borderWidth: 2,
            pointBackgroundColor: hexColor,
            pointRadius: 1,
            fill: true
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: {
            r: {
              min: 0, max: 100,
              angleLines: { color: 'rgba(255,255,255,0.05)' },
              grid: { color: 'rgba(255,255,255,0.1)' },
              ticks: { display: false },
              pointLabels: { color: '#cbd5e1', font: { size: 9 } }
            }
          },
          plugins: { legend: { display: false }, tooltip: { enabled: false } }
        }
      });
    });
  });

  // === 3. ЛОГІКА ПОРІВНЯЛЬНОГО ГРАФІКА ===
  const p1Select = document.getElementById('cmp-p1');
  const p2Select = document.getElementById('cmp-p2');
  
  // Ставимо різних гравців за замовчуванням, якщо їх більше одного
  if(G.players.length > 1) p2Select.selectedIndex = 1;

  function updateCompareChart() {
    const n1 = p1Select.value;
    const n2 = p2Select.value;
    const p1 = G.players.find(x => x.name === n1);
    const p2 = G.players.find(x => x.name === n2);
    if(!p1 || !p2) return;
    
    const prof1 = calcPlaystyleProfile(n1);
    const prof2 = calcPlaystyleProfile(n2);
    
    let rgb1 = p1.color === 'blue' ? '59,130,246' : '249,115,22';
    let hex1 = p1.color === 'blue' ? '#3b82f6' : '#f97316';
    
    let rgb2 = p2.color === 'blue' ? '59,130,246' : '249,115,22';
    let hex2 = p2.color === 'blue' ? '#3b82f6' : '#f97316';
    
    // Якщо порівнюємо гравців з однієї команди, другий колір робимо Фіолетовим
    if (rgb1 === rgb2) {
       rgb2 = '168,85,247'; 
       hex2 = '#a855f7';
    }
    
    destroyChart('compareRadar');
    const cCanvas = document.getElementById('radar-compare');
    if(!cCanvas) return;
    
    G.chartInstances['compareRadar'] = new Chart(cCanvas, {
      type: 'radar',
      data: {
        labels: ['Агресія', 'Захист', 'Механіки', 'Ротація', 'Буст', 'Командна гра'],
        datasets: [
          {
            label: n1,
            data: Object.values(prof1),
            backgroundColor: `rgba(${rgb1}, 0.3)`,
            borderColor: hex1,
            borderWidth: 2,
            pointBackgroundColor: hex1,
          },
          {
            label: n2,
            data: Object.values(prof2),
            backgroundColor: `rgba(${rgb2}, 0.3)`,
            borderColor: hex2,
            borderWidth: 2,
            pointBackgroundColor: hex2,
          }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          r: {
            min: 0, max: 100,
            angleLines: { color: 'rgba(255,255,255,0.1)' },
            grid: { color: 'rgba(255,255,255,0.1)' },
            ticks: { display: false },
            pointLabels: { color: '#f8fafc', font: { size: 13, weight: 'bold', family: "'Barlow Condensed', sans-serif" } }
          }
        },
        plugins: {
          legend: { position: 'top', labels: { color: '#fff', font: { size: 14, family: "'Space Mono', monospace" } } }
        }
      }
    });
  }

  p1Select.addEventListener('change', updateCompareChart);
  p2Select.addEventListener('change', updateCompareChart);
  requestAnimationFrame(updateCompareChart);
}

function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }
function toast(msg){ const t=document.getElementById('toast'); t.textContent=msg; t.classList.add('show'); setTimeout(()=>t.classList.remove('show'),3000); }

function showProgress(){
  document.getElementById('drop-zone').style.display='none';
  document.getElementById('parse-progress').style.display='flex';
}
function hideProgress(){
  document.getElementById('drop-zone').style.display='block';
  document.getElementById('parse-progress').style.display='none';
}

function setProgress(pct, label){
  document.getElementById('progress-fill').style.width=pct+'%';
  document.getElementById('progress-text').textContent=label;
}

function showDashboard(){
  document.getElementById('upload-screen').style.display='none';
  document.getElementById('dashboard').style.display='block';
  renderAll();
}

function goBack(){
  if (G.animState?.rafId) cancelAnimationFrame(G.animState.rafId);
  if (G.fullAnimState?.rafId) cancelAnimationFrame(G.fullAnimState.rafId);

  destroyAllCharts();

  const sel = document.getElementById('heatmap-select');
  while (sel.options.length > 3) sel.remove(3);

  document.getElementById('stats-grid').innerHTML = '';
  document.getElementById('moment-chips').innerHTML = '';
  document.getElementById('ai-analysis').style.display = 'none';
  document.getElementById('sensor-grid').innerHTML = '';
  document.getElementById('ai-probs').innerHTML = '';
  document.getElementById('ai-outcome').innerHTML = '—';
  document.getElementById('ai-verdict-box').innerHTML = '—';
  document.getElementById('match-ot-badge').innerHTML = '';
  document.getElementById('match-duration').textContent = '—';
  document.getElementById('m-time').textContent = '—';
  document.getElementById('m-touches').textContent = '—';
  document.getElementById('m-fastest').textContent = '—';
  document.getElementById('m-fastest-name').textContent = 'км/год';
  document.getElementById('m-xg').textContent = '—';
  document.getElementById('blue-goals').textContent = '0';
  document.getElementById('orange-goals').textContent = '0';
  document.getElementById('blue-players').textContent = '—';
  document.getElementById('orange-players').textContent = '—';
  document.getElementById('zone-left').textContent = '—';
  document.getElementById('zone-mid').textContent = '—';
  document.getElementById('zone-right').textContent = '—';

  const posBar = document.getElementById('positioning-bars');
  if (posBar) posBar.innerHTML = '';
  const detGrid = document.getElementById('detailed-stats-grid');
  if (detGrid) detGrid.innerHTML = '';
  ['heatmap-canvas','touch-canvas','pos-canvas','field-anim-canvas'].forEach(id => {
    const c = document.getElementById(id);
    if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height);
  });
  setOnnxStatus('', 'ONNX: INIT');

  const oldAiLabel = document.getElementById('ai-3d-label');
  if (oldAiLabel) oldAiLabel.remove();

  // НОВЕ: Очищаємо 3D сцену повністю при виході!
  scene3D = null;
  playerMeshes3D = {};
  playerNameplates = {};

  G = {
    players: [], 
    frames: [], 
    fullDf: {}, 
    hits: [], 
    ballYTimeline: [], 
    boostData: {}, 
    paths: {}, 
    matchInfo: {}, 
    doubleCommits: [],
    chartInstances: {},
    
    animState: { playing: false, frame: 0, start: 0, end: 0, rafId: null, targetFrame: 0 },
    
    fullAnimState: { playing: false, frame: 0, start: 0, end: 0, rafId: null, speed: 1 },
    
    currentActive3DContainer: 'field-anim-container',
    currentMoment: null, 
    currentAIPredict: null,
    fieldImg: G.fieldImg, 
  };
  
  document.getElementById('dashboard').style.display = 'none';
  document.getElementById('upload-screen').style.display = 'flex';
  document.getElementById('drop-zone').style.display = 'block';
  document.getElementById('parse-progress').style.display = 'none';
  document.getElementById('file-input').value = '';
  switchTab('stats');
}

function switchTab(id){
  const ids = ['stats','heatmap','boost','playstyle','replay','ai'];
  document.querySelectorAll('.nav-tab').forEach((t,i)=>{
    if(ids[i]) t.classList.toggle('active', ids[i]===id);
  });
  
  document.querySelectorAll('.section').forEach(s=>s.classList.remove('active'));
  const targetSection = document.getElementById('tab-'+id);
  if (targetSection) targetSection.classList.add('active');
  
  if(id==='heatmap') { drawHeatmap(); drawTouchMap(); }
  if(id==='playstyle') renderPlaystyleTab();
  
  // Авто-пауза при зміні вкладки
  if (id !== 'ai' && G.animState && G.animState.playing) toggleAnim();
  if (id !== 'replay' && G.fullAnimState && G.fullAnimState.playing) toggleFullAnim();

  // Логіка для 3D сцен
  if(id==='ai') {
     G.currentActive3DContainer = 'field-anim-container';
     ensure3DScene('field-anim-container');
     if (G.currentMoment) drawAnimFrame3D(G.animState.frame, G.currentMoment);
  }
  if(id==='replay') {
     startFullReplayTab();
  }
}

function destroyChart(key){
  if(G.chartInstances[key]){ G.chartInstances[key].destroy(); delete G.chartInstances[key]; }
}
function destroyAllCharts(){
  Object.keys(G.chartInstances||{}).forEach(k=>{ if(G.chartInstances[k]) G.chartInstances[k].destroy(); });
}

// Перетворює кадр реплею у точний ігровий таймер (від 5:00 до 0:00, і + для ОТ)
function formatGameTime(frame) {
  const NF = (G.fullDf?.ball?.x || []).length || 1;
  let totalSecs = G.matchInfo?.totalSecs || 300;
  const isOT = G.matchInfo?.isOvertime;
  
  // Якщо парсер не віддав час взагалі або ми знаємо, що це ОТ, але час <= 300
  if (totalSecs < 10 || (isOT && totalSecs <= 300)) {
    // Математичний хак: 1 ігрова секунда = ~33 кадри (з урахуванням мікропауз)
    totalSecs = Math.floor(NF / 33);
    // Якщо це не овертайм, жорстко фіксуємо на 300 (стандартні 5 хвилин)
    if (!isOT && totalSecs < 330) totalSecs = 300; 
  }
  
  let elapsed = Math.floor((frame / NF) * totalSecs);
  
  if (elapsed <= 300) {
    // Основний час: йде НАЗАД
    const left = 300 - Math.min(elapsed, 300);
    return `${Math.floor(left / 60)}:${(left % 60).toString().padStart(2, '0')}`;
  } else {
    // Овертайм: йде ВПЕРЕД
    const ot = Math.max(0, elapsed - 300);
    return `+${Math.floor(ot / 60)}:${(ot % 60).toString().padStart(2, '0')}`;
  }
}

// ================================================================
// ПОВНИЙ РЕПЛЕЙ МАТЧУ (FULL MATCH 3D VIEWER)
// ================================================================

function startFullReplayTab() {
  G.currentActive3DContainer = 'full-replay-container';
  ensure3DScene('full-replay-container');

  const maxFrame = (G.fullDf.ball?.x || []).length - 1;
  if (maxFrame <= 0) return;

  G.fullAnimState.start = 0;
  G.fullAnimState.end = maxFrame;
  
  if (!G.fullAnimState.playing && G.fullAnimState.frame === 0) {
     G.fullAnimState.frame = 0;
  }

  const slider = document.getElementById('full-timeline');
  if (slider) {
    slider.min = 0;
    slider.max = maxFrame;
    slider.value = G.fullAnimState.frame;
  }

  document.getElementById('full-time-current').textContent = formatGameTime(G.fullAnimState.frame);
  document.getElementById('full-time-end').textContent = formatGameTime(maxFrame);

  renderFullReplayMarkers(maxFrame);
  drawAnimFrame3D(G.fullAnimState.frame, null); 
}

function renderFullReplayMarkers(maxFrame) {
  const container = document.getElementById('full-timeline-markers');
  if (!container) return;
  container.innerHTML = '';

  G.hits.forEach(h => {
     let color = '', size = 10, z = 1;
     
     if (h.isGoal) { color = '#22c55e'; size = 16; z = 5; } // Гол
     else if (h.action === 3) { color = '#eab308'; size = 12; z = 3; } // Удар
     else if (h.action === 12) { color = '#3b82f6'; size = 12; z = 3; } // Сейв
     else if (h.action === 9) { color = '#ef4444'; size = 10; z = 2; } // Демо
     
     if (color) {
        const pct = (h.frame / maxFrame) * 100;
        const marker = document.createElement('div');
        marker.style.position = 'absolute';
        marker.style.left = `calc(${pct}% - ${size/2}px)`;
        marker.style.top = `calc(50% - ${size/2}px)`;
        marker.style.width = `${size}px`;
        marker.style.height = `${size}px`;
        marker.style.backgroundColor = color;
        marker.style.borderRadius = '50%';
        marker.style.zIndex = z;
        marker.style.boxShadow = '0 0 6px rgba(0,0,0,0.8)';
        
        // Підказка при наведенні мишки
        marker.title = `${h.isGoal ? 'ГОЛ!' : h.actionName} - ${h.player} (${formatGameTime(h.frame)})`;
        container.appendChild(marker);
     }
  });
}

function toggleFullAnim() {
  const anim = G.fullAnimState;
  if (anim.playing) {
    cancelAnimationFrame(anim.rafId);
    anim.playing = false;
    document.getElementById('full-play-btn').textContent = '▶ Грати';
    document.getElementById('full-play-btn').style.background = 'var(--blue)';
  } else {
    anim.playing = true;
    document.getElementById('full-play-btn').textContent = '⏸ Пауза';
    document.getElementById('full-play-btn').style.background = 'var(--muted)';
    
    let lastTs = performance.now();
    function tick(ts){
      const dt = ts - lastTs;
      lastTs = ts;

      // Використовуємо множник швидкості
      anim.frame += dt * 0.03 * anim.speed;

      if(anim.frame >= anim.end){ 
         anim.frame = anim.end; 
         anim.playing = false;
         document.getElementById('full-play-btn').textContent = '▶ Грати';
         document.getElementById('full-play-btn').style.background = 'var(--blue)';
      }
      
      const slider = document.getElementById('full-timeline');
      if (slider) slider.value = anim.frame;
      
      const progress = document.getElementById('full-timeline-progress');
      if (progress) progress.style.width = `${(anim.frame / anim.end) * 100}%`;
      
      const curTimeEl = document.getElementById('full-time-current');
      if (curTimeEl) curTimeEl.textContent = formatGameTime(anim.frame);

      drawAnimFrame3D(anim.frame, null);

      if(anim.playing) anim.rafId = requestAnimationFrame(tick);
    }
    anim.rafId = requestAnimationFrame(tick);
  }
}

function seekFullAnim(val) {
  const anim = G.fullAnimState;
  if (anim.playing) toggleFullAnim(); // Ставимо на паузу, якщо людина тягне повзунок
  
  anim.frame = parseFloat(val);
  
  const progress = document.getElementById('full-timeline-progress');
  if (progress) progress.style.width = `${(anim.frame / anim.end) * 100}%`;
  
  const curTimeEl = document.getElementById('full-time-current');
  if (curTimeEl) curTimeEl.textContent = formatGameTime(anim.frame);
  
  drawAnimFrame3D(anim.frame, null);
}