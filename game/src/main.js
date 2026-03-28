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
    
    let currentMode = 'human';

    // Toggle Mode Selection
    function setMode(mode, btn) {
        currentMode = mode;
        humanBtn.classList.remove('active');
        aiBtn.classList.remove('active');
        trainBtn.classList.remove('active');
        btn.classList.add('active');
    }

    humanBtn.addEventListener('click', () => setMode('human', humanBtn));
    aiBtn.addEventListener('click', () => setMode('ai', aiBtn));
    trainBtn.addEventListener('click', () => setMode('train', trainBtn));

    stopTrainingBtn.addEventListener('click', () => {
        game.stopTraining();
    });

    // Start Game
    playBtn.addEventListener('click', () => {
        initGame();
    });

    // Restart Game from GO Screen
    goScreen.addEventListener('click', () => {
        initGame();
    });

    // Space/Enter to start
    window.addEventListener('keydown', (e) => {
        if ((e.code === 'Space' || e.code === 'Enter') && (!game.active)) {
            initGame();
        }
    });

    function initGame() {
        // Clear previous players
        game.clearPlayers();
        
        if (currentMode === 'human') {
            // Add a single human player
            const controller = new HumanController();
            game.addPlayer(controller, 'human');
        } else if (currentMode === 'ai') {
            // Add a single AI player that plays the best weights
            const controller = new AIController('ai');
            game.addPlayer(controller, 'ai-best');
        } else if (currentMode === 'train') {
            // Add multiple AI players to demonstrate modularity
            for (let i = 0; i < AI_PLAYERS; i++) {
                const controller = new AIController('train');
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
