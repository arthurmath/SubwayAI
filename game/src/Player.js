/**
 * Player class representing a character in the game.
 * It has its own state, controller, and Three.js representation.
 */
class Player {
    constructor(scene, controller, id = 'player1') {
        this.id = id;
        this.scene = scene;
        this.controller = controller;

        // Visuals
        const build = buildJake();
        this.mesh = build.group;
        this.parts = build; // Save parts for animation
        this.scene.add(this.mesh);

        // Stumble / Flash state
        this.originalEmissives = new Map();

        // State
        this.reset();
    }

    reset() {
        this.lane = 1;
        this.prevLane = 1;
        this.x = LANES[1];
        this.y = 0;
        this.vy = 0;
        this.jumping = false;
        this.rolling = false;
        this.rollTimeLeft = 0;
        this.animT = 0;
        this.dead = false;
        this.score = 0;
        this.coins = 0;

        // Stumble / Flash state
        this.stumbleTimeLeft = 0;
        this.flashTimeLeft = 0;
        this.flashCount = 0;
        this.bumpTime = 0;
        this.bumpDir = 0;
        this.resetVisuals();

        // Position mesh
        this.mesh.position.set(this.x, this.y, 0);
        this.mesh.rotation.set(0, Math.PI, 0);
        this.mesh.scale.set(1, 1, 1);
        this.mesh.visible = true;
    }

    update(dt, speedMultiplier) {
        if (this.dead) return;

        // 1. Process controller inputs
        let action = this.controller.consumeAction();
        while (action) {
            switch (action) {
                case 'L':
                    if (this.lane === 0) this.stumble(-1);
                    else { this.prevLane = this.lane; this.lane = this.lane - 1; }
                    break;
                case 'R':
                    if (this.lane === 2) this.stumble(1);
                    else { this.prevLane = this.lane; this.lane = this.lane + 1; }
                    break;
                case 'J':
                    if (!this.jumping) {
                        this.jumping = true;
                        this.vy = JUMP_VELOCITY;
                        this.rolling = false;
                        this.rollTimeLeft = 0;
                    }
                    break;
                case 'S':
                    if (!this.jumping) {
                        this.rolling = true;
                        this.rollTimeLeft = ROLL_DURATION;
                    } else {
                        // Slam down if jumping
                        this.vy = Math.min(this.vy, -8);
                    }
                    break;
            }
            action = this.controller.consumeAction();
        }

        // 2. Physics / Movement
        // Lane lerping
        this.x += (LANES[this.lane] - this.x) * Math.min(1, LANE_LERP_SPEED * dt);

        // Jumping / Gravity
        if (this.jumping) {
            this.vy += GRAVITY * dt;
            this.y += this.vy * dt;
            if (this.y <= 0) {
                this.y = 0;
                this.vy = 0;
                this.jumping = false;
            }
        }

        // Rolling
        if (this.rolling) {
            this.rollTimeLeft -= dt;
            if (this.rollTimeLeft <= 0) {
                this.rolling = false;
                this.rollTimeLeft = 0;
            }
        }

        // 3. Update Mesh
        let bx = this.x;
        if (this.bumpTime > 0) {
            const bumpOffset = Math.sin((this.bumpTime / 0.2) * Math.PI) * 0.42;
            bx += this.bumpDir * bumpOffset;
        }

        this.mesh.position.set(bx, this.y, 0);
        this.animate(dt, speedMultiplier);

        // Update score
        const speed = BASE_SPEED * speedMultiplier;
        this.score += speed * dt * 0.26;

        // 4. Stumble / Flash update
        if (this.stumbleTimeLeft > 0) {
            this.stumbleTimeLeft -= dt;
        }

        if (this.bumpTime > 0) {
            this.bumpTime -= dt;
        }

        if (this.flashTimeLeft > 0) {
            this.flashTimeLeft -= dt;
            if (this.flashTimeLeft <= 0) {
                this.resetVisuals();
            } else {
                // Flash twice: 2 cycles of 0.2s (total 0.4s)
                const cycle = 0.2;
                const state = (this.flashTimeLeft % cycle) < (cycle / 2);
                this.applyRedFlash(state);
            }
        }
    }

    stumble(dir = 0) {
        if (this.dead) return;
        
        if (this.stumbleTimeLeft > 0) {
            this.triggerDeath();
        } else {
            this.stumbleTimeLeft = 5.0; // 5 seconds grace period
            this.bumpDir = dir;
            this.bumpTime = 0.2; // Quicker bump
            this.flashTimeLeft = 0.6; // Start flashing immediately (3 * 0.2s)
        }
    }

    applyRedFlash(on) {
        this.mesh.traverse(obj => {
            if (obj.isMesh && obj.material) {
                if (on) {
                    if (!this.originalEmissives.has(obj.uuid)) {
                        this.originalEmissives.set(obj.uuid, {
                            emissive: obj.material.emissive.getHex(),
                            emissiveIntensity: obj.material.emissiveIntensity
                        });
                    }
                    obj.material.emissive.setHex(0xff0000);
                    obj.material.emissiveIntensity = 1.0;
                } else {
                    this.restoreMaterial(obj);
                }
            }
        });
    }

    restoreMaterial(obj) {
        if (this.originalEmissives.has(obj.uuid)) {
            const orig = this.originalEmissives.get(obj.uuid);
            obj.material.emissive.setHex(orig.emissive);
            obj.material.emissiveIntensity = orig.emissiveIntensity;
        }
    }

    resetVisuals() {
        this.mesh.traverse(obj => {
            if (obj.isMesh && obj.material) {
                this.restoreMaterial(obj);
            }
        });
        this.originalEmissives.clear();
    }

    animate(dt, speedMultiplier) {
        const speed = BASE_SPEED * speedMultiplier;
        this.animT += dt * (speed / BASE_SPEED) * 1.65;
        const t = this.animT;
        const rf = 8.2; // Running frequency (faster)
        const ra = 0.58; // Running amplitude (more amplitude)

        if (this.rolling) {
            // Tackle slide (football style): flattening backwards, feet first
            // We use a positive rotation on X because the player is already rotated 180° on Y
            this.mesh.rotation.x = Math.PI / 2.2; 
            this.mesh.scale.set(1, 1, 1); // Reset scale to avoid distortion
            
            // Adjust height: since the pivot is at the feet, rotating 90° puts the body on the floor
            this.mesh.position.y = this.y + 0.22;
            
            // Pose limbs for a dynamic tackle
            this.parts.llG.rotation.x = -0.15; // Lowered forward leg
            this.parts.rlG.rotation.x = 0.7;   // Leg tucked in
            
            this.parts.laG.rotation.x = 0.4; // Arms slightly back for support
            this.parts.raG.rotation.x = 0.4;
            this.parts.laG.rotation.z = -0.2;
            this.parts.raG.rotation.z = 0.2;

            this.mesh.rotation.z *= 0.5; // Reduce lane-change tilt during slide
        } else {
            this.mesh.rotation.x = 0; // Reset X rotation
            this.mesh.position.y = this.y;
            this.mesh.scale.set(1, 1, 1);
            if (this.parts.torso) this.parts.torso.position.y = 0.87 + Math.abs(Math.sin(t * rf)) * 0.04;
            this.parts.llG.rotation.x = Math.sin(t * rf) * ra;
            this.parts.rlG.rotation.x = -Math.sin(t * rf) * ra;
            this.parts.laG.rotation.x = -Math.sin(t * rf) * ra * 0.85;
            this.parts.raG.rotation.x = Math.sin(t * rf) * ra * 0.85;
            this.parts.laG.rotation.z = 0;
            this.parts.raG.rotation.z = 0;

            if (this.jumping) {
                const tuck = Math.max(0, Math.sin(Math.max(0, this.y / 2.5) * Math.PI));
                // One leg forward, one backward (more natural)
                this.parts.llG.rotation.x = -tuck * 1.4;
                this.parts.rlG.rotation.x = tuck * 0.8;
                // Arms in opposition to legs
                this.parts.laG.rotation.x = tuck * 1.1; 
                this.parts.raG.rotation.x = -tuck * 1.1;
            }

            // Tilt when switching lanes
            const dx = LANES[this.lane] - this.x;
            this.mesh.rotation.z += (-dx * 0.16 - this.mesh.rotation.z) * 0.18;
        }

        // Shoe bobbing is now handled by leg hierarchy
    }

    triggerDeath() {
        this.dead = true;
        this.resetVisuals();
        // Death animation can be added here
    }
}
