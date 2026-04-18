/**
 * Main entry point for the Subway AI game.
 */

document.addEventListener('DOMContentLoaded', () => {
    const game = new Game();
    
    const playBtn = document.getElementById('play-btn');
    const humanBtn = document.getElementById('human-mode-btn');
    const aiBtn = document.getElementById('ai-mode-btn');
    const trainBtn = document.getElementById('train-mode-btn');
    const stopTrainingBtn = document.getElementById('stop-training-btn');
    const goScreen = document.getElementById('go-screen');
    const trainSubBtns = document.getElementById('train-sub-btns');
    const coldStartBtn = document.getElementById('cold-start-btn');
    const warmStartBtn = document.getElementById('warm-start-btn');
    const aiWeightsList = document.getElementById('ai-weights-list');
    const warmWeightsList = document.getElementById('warm-weights-list');
    const goRestartBtn = document.getElementById('go-restart-btn');
    const goHomeBtn = document.getElementById('go-home-btn');
    
    let currentMode = 'human';
    let currentGameId = 0;
    let warmStart = false;
    let selectedWeightsFile = null;     // null = best (AI Player mode)
    let selectedWarmWeightsFile = null; // null = best (Warm Start mode)

    // Toggle Mode Selection
    function setMode(mode, btn) {
        currentMode = mode;
        humanBtn.classList.remove('active');
        aiBtn.classList.remove('active');
        trainBtn.classList.remove('active');
        btn.classList.add('active');
        trainSubBtns.classList.toggle('hidden', mode !== 'train');
        aiWeightsList.classList.toggle('hidden', mode !== 'ai');
        if (mode === 'ai') loadWeightsList(aiWeightsList, f => { selectedWeightsFile = f; });
    }

    function loadWeightsList(container, onSelect) {
        container.innerHTML = '<div style="color:rgba(255,255,255,0.4);font-size:12px;padding:6px;">Querying server…</div>';
        onSelect(null);
        const ws = new WebSocket('ws://127.0.0.1:8765');
        ws.onopen = () => ws.send(JSON.stringify({ type: 'query_weights_list' }));
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            ws.close();
            buildWeightsList(container, data.weights || [], onSelect);
        };
        ws.onerror = () => {
            container.innerHTML = '<div style="color:rgba(255,100,100,0.8);font-size:12px;padding:6px;">Server not reachable</div>';
        };
    }

    function buildWeightsList(container, weights, onSelect) {
        container.innerHTML = '';
        if (weights.length === 0) {
            container.innerHTML = '<div style="color:rgba(255,255,255,0.4);font-size:12px;padding:6px;">No weights found</div>';
            return;
        }
        weights.forEach((w, i) => {
            const item = document.createElement('button');
            item.className = 'weights-item' + (i === 0 ? ' selected' : '');
            item.innerHTML = `<span class="weights-item-score">${w.score}m</span><span>${w.date}${i === 0 ? '&nbsp;·&nbsp;⭐ Best' : ''}</span><span class="weights-item-name">${w.filename}</span>`;
            item.addEventListener('click', () => {
                container.querySelectorAll('.weights-item').forEach(el => el.classList.remove('selected'));
                item.classList.add('selected');
                onSelect(i === 0 ? null : w.filename);
            });
            container.appendChild(item);
        });
    }

    function setTrainSubMode(isWarm) {
        warmStart = isWarm;
        coldStartBtn.classList.toggle('selected', !isWarm);
        warmStartBtn.classList.toggle('selected', isWarm);
        warmWeightsList.classList.toggle('hidden', !isWarm);
        if (isWarm) {
            loadWeightsList(warmWeightsList, f => { selectedWarmWeightsFile = f; });
        }
    }

    humanBtn.addEventListener('click', () => setMode('human', humanBtn));
    aiBtn.addEventListener('click', () => setMode('ai', aiBtn));
    trainBtn.addEventListener('click', () => setMode('train', trainBtn));

    coldStartBtn.addEventListener('click', () => setTrainSubMode(false));
    warmStartBtn.addEventListener('click', () => setTrainSubMode(true));

    stopTrainingBtn.addEventListener('click', () => {
        game.stopTraining();
    });

    // Start Game
    playBtn.addEventListener('click', () => {
        initGame();
    });

    // Restart Game from GO Screen
    goScreen.addEventListener('click', (e) => {
        // Only trigger if we didn't click a button
        if (e.target.classList.contains('mode-btn')) return;
        initGame();
    });

    if (goRestartBtn) {
        goRestartBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            initGame();
        });
    }

    if (goHomeBtn) {
        goHomeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            goScreen.classList.add('hidden');
            document.getElementById('start-screen').classList.remove('hidden');
            game.clearPlayers();
        });
    }

    // Space/Enter to start
    window.addEventListener('keydown', (e) => {
        if ((e.code === 'Space' || e.code === 'Enter') && (!game.active)) {
            initGame();
        }
        if (e.code === 'KeyP' || e.code === 'Escape') {
            game.togglePause();
        }
    });

    function initGame() {
        currentGameId++;
        
        // Clear previous players
        game.clearPlayers();
        
        if (currentMode === 'human') {
            const controller = new HumanController();
            game.addPlayer(controller, 'human');
        } else if (currentMode === 'ai') {
            const controller = new AIController('ai', currentGameId, 0, false, selectedWeightsFile);
            game.addPlayer(controller, 'ai-best');
        } else if (currentMode === 'train') {
            for (let i = 0; i < AI_PLAYERS; i++) {
                const controller = new AIController('train', currentGameId, i * 35, warmStart, warmStart ? selectedWarmWeightsFile : null);
                game.addPlayer(controller, `ai-train-${i}`);
            }
        }

        if (currentMode === 'train') {
            stopTrainingBtn.classList.remove('hidden');
        } else {
            stopTrainingBtn.classList.add('hidden');
        }

        game.mode = currentMode;
        game.start();
    }
});
