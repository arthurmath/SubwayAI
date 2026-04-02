/**
 * Game class manages the game loop, players, obstacles, and coins.
 */
class Game {
    constructor() {
        this.world = new World('game-wrap');
        this.players = [];
        this.obstacles = [];
        this.coins = [];
        this.active = false;
        this.speedMultiplier = 1.0;
        this.lastSpeedUpScore = 0;
        this.time = 0;
        this.paused = false;

        this.setupPools();
        this.setupUI();
    }

    setupPools() {
        // Obstacle Pool
        for (let i = 0; i < 12; i++) this.obstacles.push({ mesh: buildTrain(), type: 'train', active: false });
        for (let i = 0; i < 8; i++) this.obstacles.push({ mesh: buildLowBarrier(), type: 'low', active: false });
        for (let i = 0; i < 8; i++) this.obstacles.push({ mesh: buildHighFence(), type: 'high', active: false });
        
        this.obstacles.forEach(o => {
            o.mesh.visible = false;
            this.world.scene.add(o.mesh);
        });

        // Coin Pool
        for (let i = 0; i < 90; i++) {
            const mesh = buildCoin();
            mesh.visible = false;
            this.world.scene.add(mesh);
            this.coins.push({ mesh, active: false, lane: 0, z: 0, baseY: 0.52 });
        }
    }

    setupUI() {
        this.scoreEl = document.getElementById('score-val');
        this.coinEl = document.getElementById('coin-val');
        this.aliveCountEl = document.getElementById('alive-count');
        this.flashEl = document.getElementById('speed-flash');
        this.goScreen = document.getElementById('go-screen');
        this.paramsContainer = document.getElementById('ai-params-container');
        this.statsContainer = document.getElementById('ai-stats');
        this.statIterationEl = document.getElementById('stat-iteration');
        this.statTrainCountEl = document.getElementById('stat-train-count');
        this.statBestDistEl = document.getElementById('stat-best-dist');
        this.statAvgScoreEl = document.getElementById('stat-avg-score');
        this.pauseOverlay = document.getElementById('pause-overlay');
        this.paramValueEls = {};
        this.humanRewardEl = null;
        this.lastHumanScore = 0;
        this.lastHumanCoins = 0;
        this.totalHumanReward = 0;
    }

    addPlayer(controller, id) {
        const player = new Player(this.world.scene, controller, id);
        this.players.push(player);
        return player;
    }

    clearPlayers() {
        this.players.forEach(p => {
            if (p.mesh) this.world.scene.remove(p.mesh);
            if (p.controller && typeof p.controller.destroy === 'function') {
                p.controller.destroy();
            }
        });
        this.players = [];
    }

    start() {
        this.active = true;
        this.paused = false;
        if (this.pauseOverlay) this.pauseOverlay.classList.add('hidden');
        this.speedMultiplier = 1.0;
        this.lastSpeedUpScore = 0;
        this.time = 0;
        this.lastHumanScore = 0;
        this.lastHumanCoins = 0;
        this.totalHumanReward = 0;
        
        if (this.mode === 'train' && this.aliveCountEl) {
            this.aliveCountEl.classList.remove('hidden');
            if (this.statsContainer) this.statsContainer.classList.remove('hidden');
        } else if (this.aliveCountEl) {
            this.aliveCountEl.classList.add('hidden');
            if (this.statsContainer) this.statsContainer.classList.add('hidden');
        }
        
        this.players.forEach(p => p.reset());
        this.clearEnvironment();
        this.seedWorld();
        
        this.goScreen.classList.add('hidden');
        document.getElementById('start-screen').classList.add('hidden');
        
        this.lastTimestamp = performance.now();
        requestAnimationFrame((t) => this.loop(t));
    }

    clearEnvironment() {
        this.obstacles.forEach(o => { o.active = false; o.mesh.visible = false; });
        this.coins.forEach(c => { c.active = false; c.mesh.visible = false; });
    }

    seedWorld() {
        for (let i = 0; i < 6; i++) this.spawnCluster(-22 - i * 16);
        for (let i = 0; i < 7; i++) this.spawnCoins(-7 - i * 10);
    }

    spawnCluster(z) {
        const availableLanes = [0, 1, 2];
        this.shuffle(availableLanes);
        const count = Math.random() < 0.52 ? 1 : 2;

        for (let i = 0; i < count; i++) {
            const type = this.pickRandomType();
            const obs = this.obstacles.find(o => !o.active && o.type === type) || this.obstacles.find(o => !o.active);
            if (!obs) continue;

            obs.active = true;
            obs.lane = availableLanes[i];
            obs.mesh.position.set(LANES[obs.lane], 0, z);
            obs.mesh.visible = true;
        }
    }

    spawnCoins(z) {
        const pattern = Math.floor(Math.random() * 5);
        const lane = Math.floor(Math.random() * 3);
        const count = 5 + Math.floor(this.speedMultiplier * 2);

        if (pattern === 0) { // Straight line
            for (let i = 0; i < count; i++) this.activateCoin(lane, z - i * 1.25, 0.52);
        } else if (pattern === 1) { // Zig zag
            const seq = [0, 1, 2, 1, 0, 1, 2];
            for (let i = 0; i < seq.length; i++) this.activateCoin(seq[i], z - i * 1.0, 0.52);
        } else { // Generic arc or other
            for (let i = 0; i < 7; i++) {
                const y = 0.55 + Math.sin((i / 6) * Math.PI) * 2.1;
                this.activateCoin(lane, z - i * 1.4, y);
            }
        }
    }

    activateCoin(lane, z, y) {
        const coin = this.coins.find(c => !c.active);
        if (!coin) return;
        coin.active = true;
        coin.lane = lane;
        coin.z = z;
        coin.baseY = y;
        coin.mesh.position.set(LANES[lane], y, z);
        coin.mesh.visible = true;
    }

    pickRandomType() {
        const r = Math.random();
        return r < 0.44 ? 'train' : r < 0.70 ? 'high' : 'low';
    }

    shuffle(a) {
        for (let i = a.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [a[i], a[j]] = [a[j], a[i]];
        }
    }

    loop(timestamp) {
        if (!this.active) return;

        if (this.paused) {
            this.lastTimestamp = timestamp;
            requestAnimationFrame((t) => this.loop(t));
            return;
        }

        const dt = Math.min((timestamp - this.lastTimestamp) / 1000, 0.05);
        this.lastTimestamp = timestamp;
        this.time += dt;

        const speed = BASE_SPEED * this.speedMultiplier;

        // Update World & Players
        this.world.update(dt, this.speedMultiplier);
        
        let allDead = true;
        let maxScore = 0;
        let totalCoins = 0;

        const gameState = {
            obstacles: this.obstacles.filter(o => o.active).map(o => ({ lane: o.lane, z: o.mesh.position.z, type: o.type })),
            coins: this.coins.filter(c => c.active).map(c => ({ lane: c.lane, z: c.z })),
            speed: speed
        };

        // UI AI Parameters for Human Mode
        if (this.mode === 'human' && this.paramsContainer) {
            this.paramsContainer.classList.remove('hidden');
            const humanPlayer = this.players.find(p => !p.dead);
            if (humanPlayer) {
                const aiState = this.extractAIState({
                    player: { lane: humanPlayer.lane, x: humanPlayer.x, y: humanPlayer.y, vy: humanPlayer.vy, rolling: humanPlayer.rolling, dead: humanPlayer.dead, score: humanPlayer.score, coins: humanPlayer.coins },
                    ...gameState
                });
                this.updateParamsDisplay(aiState);

                const scoreDelta = humanPlayer.score - this.lastHumanScore;
                const coinsDelta = humanPlayer.coins - this.lastHumanCoins;
                this.totalHumanReward += scoreDelta * 5.0 + coinsDelta * 0.5;
                this.lastHumanScore = humanPlayer.score;
                this.lastHumanCoins = humanPlayer.coins;
                if (this.humanRewardEl) {
                    this.humanRewardEl.textContent = this.totalHumanReward.toFixed(1);
                }
            }
        } else if (this.paramsContainer) {
            this.paramsContainer.classList.add('hidden');
        }

        this.players.forEach(p => {
            if (p.controller.update && typeof p.controller.update === 'function') {
                p.controller.update({
                    player: { lane: p.lane, x: p.x, y: p.y, vy: p.vy, dead: p.dead, score: p.score, coins: p.coins },
                    ...gameState
                });
            }

            p.update(dt, this.speedMultiplier);
            if (!p.dead) {
                allDead = false;
                this.checkCollisions(p);
            }
            maxScore = Math.max(maxScore, p.score);
            totalCoins += p.coins;
        });

        if (allDead) {
            this.gameOver(maxScore, totalCoins);
            return;
        }

        // Speed Up Logic
        if (maxScore - this.lastSpeedUpScore >= SCORE_LEVEL_FOR_SPEEDUP) {
            this.lastSpeedUpScore = maxScore;
            this.speedMultiplier = Math.min(this.speedMultiplier + SPEED_STEP, MAX_SPEED_MULT);
            this.showFlash();
        }

        // Update Obstacles & Coins
        this.updateObstacles(dt, speed);
        this.updateCoins(dt, speed);
        this.checkSpawning();

        // UI Updates
        this.scoreEl.textContent = Math.floor(maxScore) + 'm';
        this.coinEl.textContent = totalCoins;

        if (this.mode === 'train' && this.aliveCountEl) {
            const alivePlayers = this.players.filter(p => !p.dead);
            const aliveCount = alivePlayers.length;
            this.aliveCountEl.textContent = `Agents: ${aliveCount}/${this.players.length}`;
            
            // Find any player that has stats (prioritize alive ones)
            let statsSource = alivePlayers.find(p => p.controller && p.controller.stats);
            if (!statsSource) {
                statsSource = this.players.find(p => p.controller && p.controller.stats);
            }

            if (statsSource && statsSource.controller.stats) {
                const stats = statsSource.controller.stats;
                if (this.statIterationEl) this.statIterationEl.textContent = stats.iteration;
                if (this.statTrainCountEl) this.statTrainCountEl.textContent = stats.trainCount;
                if (this.statBestDistEl) this.statBestDistEl.textContent = Math.floor(stats.bestScore) + 'm';
                if (this.statAvgScoreEl) this.statAvgScoreEl.textContent = stats.avgScore.toFixed(1) + 'm';
            }
        }

        // Camera follow (simple average of players)
        this.updateCamera();

        this.world.render();
        requestAnimationFrame((t) => this.loop(t));
    }

    updateObstacles(dt, speed) {
        this.obstacles.forEach(o => {
            if (!o.active) return;
            o.mesh.position.z += speed * dt;
            if (o.mesh.position.z > RECYCLE_Z) {
                o.active = false;
                o.mesh.visible = false;
            }
        });
    }

    updateCoins(dt, speed) {
        this.coins.forEach(c => {
            if (!c.active) return;
            c.z += speed * dt;
            c.mesh.position.z = c.z;
            c.mesh.position.y = c.baseY + Math.sin(this.time * 3.8 + c.z * 0.4) * 0.07;
            c.mesh.rotation.y += dt * 3.2;
            if (c.z > RECYCLE_Z) {
                c.active = false;
                c.mesh.visible = false;
            }
        });
    }

    checkSpawning() {
        let maxObsZ = 0;
        this.obstacles.forEach(o => { if (o.active && o.mesh.position.z < maxObsZ) maxObsZ = o.mesh.position.z; });
        if (maxObsZ > -44) this.spawnCluster(maxObsZ - (13 + Math.random() * 9));

        let maxCoinZ = 0;
        this.coins.forEach(c => { if (c.active && c.z < maxCoinZ) maxCoinZ = c.z; });
        if (maxCoinZ > -40) this.spawnCoins(maxCoinZ - (8 + Math.random() * 5));
    }

    checkCollisions(player) {
        const px = player.x;
        const py = player.y;

        // Obstacles
        for (const obs of this.obstacles) {
            if (!obs.active) continue;
            const oz = obs.mesh.position.z;
            const hzEnter = (obs.type === 'low' || obs.type === 'high') ? -1.0 : HIT_Z_ENTER;
            const hzExit = (obs.type === 'low' || obs.type === 'high') ? 1.0 : HIT_Z_EXIT;
            if (oz < hzEnter || oz > hzExit) continue;

            const hx = obs.type === 'train' ? HIT_X_TRAIN : HIT_X_BARRIER;
            if (Math.abs(px - LANES[obs.lane]) > hx) continue;

            if (obs.type === 'train') {
                if (py < 1.88) {
                    if (oz > -1.2) { // side hit
                        const direction = player.lane - player.prevLane;
                        player.stumble(direction);
                        player.lane = player.prevLane;
                        return;
                    } else {
                        player.triggerDeath();
                        return;
                    }
                }
            } else if (obs.type === 'low') {
                if (py < 0.70) { player.triggerDeath(); return; }
            } else if (obs.type === 'high') {
                const playerTop = py + (player.rolling ? 0.81 : 1.5);
                if (playerTop > 0.85 && py < 2.30) { player.triggerDeath(); return; }
            }
        }

        // Coins
        for (const coin of this.coins) {
            if (!coin.active) continue;
            if (coin.z < -1.55 || coin.z > 1.55) continue;
            if (Math.abs(px - LANES[coin.lane]) > 0.92) continue;
            if (Math.abs((py + 0.85) - coin.baseY) > 1.0) continue;

            // Collect
            coin.active = false;
            coin.mesh.visible = false;
            player.coins++;
        }
    }

    updateCamera() {
        // Find center of all alive players
        let activePlayers = this.players.filter(p => !p.dead);
        if (activePlayers.length === 0) return;

        let avgX = activePlayers.reduce((sum, p) => sum + p.x, 0) / activePlayers.length;
        
        // Soft follow
        this.world.camera.position.x += (avgX * 0.14 - this.world.camera.position.x) * 0.07;
        this.world.camera.lookAt(new THREE.Vector3(avgX * 0.05, 0.8, -10));
    }

    showFlash() {
        this.flashEl.style.opacity = '1';
        setTimeout(() => this.flashEl.style.opacity = '0', 1400);
    }

    gameOver(score, coins) {
        this.active = false;
        this.paused = false;
        if (this.pauseOverlay) this.pauseOverlay.classList.add('hidden');
        document.getElementById('go-score').textContent = Math.floor(score) + 'm';
        document.getElementById('go-coins-earned').textContent = '+' + coins;
        this.goScreen.classList.remove('hidden');
        
        if (this.mode === 'train') {
            setTimeout(() => {
                this.goScreen.click();
            }, 1000);
        }
    }

    stopTraining() {
        if (this.mode !== 'train') return;

        let aiPlayer = this.players.find(p => p.id.startsWith('ai-train'));
        if (aiPlayer && aiPlayer.controller && typeof aiPlayer.controller.saveWeights === 'function') {
            aiPlayer.controller.saveWeights();
        }

        this.active = false;
        this.paused = false;
        if (this.pauseOverlay) this.pauseOverlay.classList.add('hidden');
        this.clearPlayers();
        
        document.getElementById('start-screen').classList.remove('hidden');
        document.getElementById('stop-training-btn').classList.add('hidden');
        if (this.aliveCountEl) this.aliveCountEl.classList.add('hidden');
        if (this.statsContainer) this.statsContainer.classList.add('hidden');
        this.goScreen.classList.add('hidden');
    }

    extractAIState(gameState) {
        const player = gameState.player || {};
        const obstacles = gameState.obstacles || [];
        const coins = gameState.coins || [];
        const speed = gameState.speed || 10.0;
        
        const state = {};
        
        // 1. Player (matches extract_state in main.py)
        state['Lane'] = (player.lane ?? 1) - 1.0;
        state['Y'] = (player.y ?? 0) / 3.0;
        state['Sliding'] = player.rolling ? 1.0 : 0.0;
        state['Speed'] = speed / 10.0;
        
        // 2. Obstacles: Next per lane
        const obs_by_lane = {};
        for (let lane = 0; lane < 3; lane++) {
            const laneId = lane + 1;
            const obs_in_lane = obstacles
                .filter(o => o.lane === lane && (o.z ?? 0) < 0)
                .sort((a, b) => (b.z ?? 0) - (a.z ?? 0));
            
            if (obs_in_lane.length > 0) {
                const obs = obs_in_lane[0];
                obs_by_lane[lane] = obs.z ?? 0;
                state[`L${laneId}_Z`] = Math.abs(obs_by_lane[lane]) / 50.0;
                const t = obs.type;
                let otype = 1.0; // train default
                if (t === 'low') otype = 0.0;
                else if (t === 'high') otype = 0.5;
                state[`L${laneId}_T`] = otype;
            } else {
                obs_by_lane[lane] = -500.0;
                state[`L${laneId}_Z`] = 1.0;
                state[`L${laneId}_T`] = -1.0;
            }
        }
        
        // 3. Coins: Next per lane and Count BEFORE next obstacle
        for (let lane = 0; lane < 3; lane++) {
            const laneId = lane + 1;
            const coins_in_lane = coins
                .filter(c => c.lane === lane && (c.z ?? 0) < 0)
                .sort((a, b) => (b.z ?? 0) - (a.z ?? 0));
            
            const obs_z = obs_by_lane[lane];
            const coins_before_obs = coins_in_lane.filter(c => (c.z ?? 0) > obs_z);
            
            if (coins_before_obs.length > 0) {
                const first_coin = coins_before_obs[0];
                state[`C${laneId}_Z`] = Math.abs(first_coin.z ?? 0) / 50.0;
                state[`C${laneId}_N`] = coins_before_obs.length;
            } else {
                state[`C${laneId}_Z`] = 1.0;
                state[`C${laneId}_N`] = 0.0;
            }
        }
        
        return state;
    }

    updateParamsDisplay(state) {
        if (!this.paramsContainer) return;

        // If structure is not created yet
        if (Object.keys(this.paramValueEls).length === 0) {
            const layout = [
                ['Lane', 'Y', 'L1_Z', 'L2_Z', 'L3_Z', 'C1_Z', 'C2_Z', 'C3_Z'],
                ['Speed', 'Sliding', 'L1_T', 'L2_T', 'L3_T', 'C1_N', 'C2_N', 'C3_N']
            ];

            layout.forEach(rowKeys => {
                const row = document.createElement('div');
                row.className = 'param-row';
                rowKeys.forEach(key => {
                    const item = document.createElement('div');
                    item.className = 'param-item';
                    const name = document.createElement('div');
                    name.className = 'param-name';
                    name.textContent = key;
                    const val = document.createElement('div');
                    val.className = 'param-value';
                    this.paramValueEls[key] = val;
                    item.appendChild(name);
                    item.appendChild(val);
                    row.appendChild(item);
                });
                this.paramsContainer.appendChild(row);
            });

            const rewardRow = document.createElement('div');
            rewardRow.className = 'param-reward-row';
            const rewardLabel = document.createElement('div');
            rewardLabel.className = 'param-reward-label';
            rewardLabel.textContent = 'REWARD';
            const rewardVal = document.createElement('div');
            rewardVal.className = 'param-reward-value';
            rewardVal.textContent = '0.0';
            this.humanRewardEl = rewardVal;
            rewardRow.appendChild(rewardLabel);
            rewardRow.appendChild(rewardVal);
            this.paramsContainer.appendChild(rewardRow);
        }
        
        // Update values
        for (const [key, value] of Object.entries(state)) {
            if (this.paramValueEls[key]) {
                this.paramValueEls[key].textContent = value.toFixed(2);
            }
        }
    }

    togglePause() {
        if (!this.active) return;
        this.paused = !this.paused;
        if (this.pauseOverlay) {
            if (this.paused) {
                this.pauseOverlay.classList.remove('hidden');
            } else {
                this.pauseOverlay.classList.add('hidden');
            }
        }
    }
}
