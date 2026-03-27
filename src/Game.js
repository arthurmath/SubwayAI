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
        this.flashEl = document.getElementById('speed-flash');
        this.goScreen = document.getElementById('go-screen');
    }

    addPlayer(controller, id) {
        const player = new Player(this.world.scene, controller, id);
        this.players.push(player);
        return player;
    }

    clearPlayers() {
        this.players.forEach(p => {
            if (p.mesh) this.world.scene.remove(p.mesh);
        });
        this.players = [];
    }

    start() {
        this.active = true;
        this.speedMultiplier = 1.0;
        this.lastSpeedUpScore = 0;
        this.time = 0;
        
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
            for (let i = 0; i < 5; i++) {
                const y = 0.55 + Math.sin((i / 4) * Math.PI) * 1.2;
                this.activateCoin(lane, z - i * 0.9, y);
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

        const dt = Math.min((timestamp - this.lastTimestamp) / 1000, 0.05);
        this.lastTimestamp = timestamp;
        this.time += dt;

        const speed = BASE_SPEED * this.speedMultiplier;

        // Update World & Players
        this.world.update(dt, this.speedMultiplier);
        
        let allDead = true;
        let maxScore = 0;
        let totalCoins = 0;

        this.players.forEach(p => {
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
                if (!player.rolling && py < 0.70) { player.triggerDeath(); return; }
            } else if (obs.type === 'high') {
                if (py < 1.45) { player.triggerDeath(); return; }
            }
        }

        // Coins
        for (const coin of this.coins) {
            if (!coin.active) continue;
            if (coin.z < -1.55 || coin.z > 1.55) continue;
            if (Math.abs(px - LANES[coin.lane]) > 0.92) continue;
            if (Math.abs(py - coin.baseY) > 0.82) continue;

            // Collect
            coin.active = false;
            coin.mesh.visible = false;
            player.coins++;
            player.score += 10;
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
        document.getElementById('go-score').textContent = Math.floor(score) + 'm';
        document.getElementById('go-coins-earned').textContent = '+' + coins + ' 🪙';
        this.goScreen.classList.remove('hidden');
    }
}
