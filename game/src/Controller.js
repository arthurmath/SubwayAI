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

/**
 * Placeholder for AI Controller.
 * In the future, this will be connected to a Reinforcement Learning agent.
 */
class AIController extends Controller {
    constructor() {
        super();
        this.socket = new WebSocket('ws://localhost:8765');
        this.waitingForAction = false;

        this.socket.onopen = () => {
            console.log('Connected to AI Server');
        };

        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.action) {
                this.addAction(data.action);
            }
            this.waitingForAction = false;
        };

        this.socket.onclose = () => {
            console.log('Disconnected from AI Server');
        };
    }

    // This method can be called by the RL agent to decide on actions
    update(gameState) {
        if (!this.waitingForAction && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify(gameState));
            this.waitingForAction = true;
        }
    }
}
