/**
 * Controller abstraction to allow both human and AI to control characters.
 */

class Controller {
    constructor() {
        this.actionQueue = [];
    }

    // Get the next action and clear it
    consumeAction() {
        return this.actionQueue.shift();
    }

    // Add an action to the queue: 'L' (Left), 'R' (Right), 'J' (Jump), 'S' (Slide)
    addAction(action) {
        this.actionQueue.push(action);
    }
}

class HumanController extends Controller {
    constructor() {
        super();
        this.setupListeners();
    }

    setupListeners() {
        window.addEventListener('keydown', (e) => {
            if (e.repeat) return;
            switch (e.code) {
                case 'ArrowLeft': this.addAction('L'); break;
                case 'ArrowRight': this.addAction('R'); break;
                case 'ArrowDown': this.addAction('S'); break;
                case 'ArrowUp': this.addAction('J'); e.preventDefault(); break;
            }
        });

        // Touch controls
        let txS = 0, tyS = 0, ttS = 0;
        window.addEventListener('touchstart', (e) => {
            txS = e.touches[0].clientX;
            tyS = e.touches[0].clientY;
            ttS = Date.now();
        }, { passive: true });

        window.addEventListener('touchend', (e) => {
            const t = e.changedTouches[0];
            const dx = t.clientX - txS;
            const dy = t.clientY - tyS;
            const dt = Date.now() - ttS;

            if (dt > 450 || (Math.abs(dx) < 12 && Math.abs(dy) < 12)) return;
            
            if (Math.abs(dx) > Math.abs(dy)) {
                this.addAction(dx > 0 ? 'R' : 'L');
            } else {
                this.addAction(dy < 0 ? 'J' : 'S');
            }
        }, { passive: true });
    }
}


class AIController extends Controller {
    constructor(mode = 'ai', gameId = 0, connectionDelay = 0, warmStart = false, weightsFile = null) {
        super();
        this.mode = mode;
        this.gameId = gameId;
        this.warmStart = warmStart;
        this.weightsFile = weightsFile;
        this.waitingForAction = false;
        this.initialized = false;
        this.lastActionTime = 0;

        const connect = () => {
            this.socket = new WebSocket('ws://127.0.0.1:8765');
            
            this.socket.onopen = () => {
                console.log('Connected to AI Server in ' + this.mode + ' mode' + (this.warmStart ? ' (warm start)' : ''));
                this.socket.send(JSON.stringify({ type: 'init', mode: this.mode, game_id: this.gameId, warm_start: this.warmStart, weights_file: this.weightsFile }));
                this.initialized = true;
            };

            this.socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.action) {
                    this.addAction(data.action);
                }
                if (data.iteration !== undefined) {
                    this.stats = {
                        iteration: data.iteration,
                        trainCount: data.train_count,
                        bestScore: data.best_score,
                        reward: data.reward,
                        avgScore: data.avg_score,
                        probs: data.probs ?? null
                    };
                }
                this.waitingForAction = false;
            };

            this.socket.onclose = () => {
                console.log('Disconnected from AI Server');
                this.initialized = false;
            };
        };

        if (connectionDelay > 0) {
            setTimeout(connect, connectionDelay);
        } else {
            connect();
        }
    }

    // This method can be called by the RL agent to decide on actions
    update(gameState) {
        if (!this.initialized || !this.socket) return;
        const now = Date.now();
        if (!this.waitingForAction
            && this.socket.readyState === WebSocket.OPEN
            && now - this.lastActionTime >= AI_ACTION_COOLDOWN_MS) {
            this.socket.send(JSON.stringify({ type: 'state', data: gameState }));
            this.waitingForAction = true;
            this.lastActionTime = now;
        }
    }

    saveWeights() {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify({ type: 'save' }));
        }
    }

    destroy() {
        if (this.socket) {
            this.socket.close();
        }
    }
}
