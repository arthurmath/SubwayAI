/**
 * Main entry point for the Subway AI game.
 */

document.addEventListener('DOMContentLoaded', () => {
    const game = new Game();
    
    const playBtn = document.getElementById('play-btn');
    const humanBtn = document.getElementById('human-mode-btn');
    const aiBtn = document.getElementById('ai-mode-btn');
    const goScreen = document.getElementById('go-screen');
    
    let currentMode = 'human';

    // Toggle Mode Selection
    humanBtn.addEventListener('click', () => {
        currentMode = 'human';
        humanBtn.classList.add('active');
        aiBtn.classList.remove('active');
    });

    aiBtn.addEventListener('click', () => {
        currentMode = 'ai';
        aiBtn.classList.add('active');
        humanBtn.classList.remove('active');
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
        } else {
            // Add multiple AI players to demonstrate modularity
            // In the future, each will have its own RL model instance
            for (let i = 0; i < AI_PLAYERS; i++) {
                const controller = new AIController();
                game.addPlayer(controller, `ai-${i}`);
            }
        }

        game.start();
    }
});
