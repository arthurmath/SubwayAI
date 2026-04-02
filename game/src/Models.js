/**
 * Geometry builders for characters, obstacles, and world elements.
 */

// Helper: Quick box creation
function createBox(w, h, d, color, options = {}) {
    const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(w, h, d),
        new THREE.MeshStandardMaterial({
            color,
            roughness: options.roughness || 0.7,
            metalness: options.metalness || 0,
            emissive: options.emissive || 0,
            emissiveIntensity: options.emissiveIntensity || 0
        })
    );
    mesh.castShadow = true;
    mesh.receiveShadow = !!options.receiveShadow;
    return mesh;
}

// Player Arm Builder
function addArm(group, xPos, sleeveCol) {
    const ag = new THREE.Group();
    ag.position.set(xPos, 1.05, 0);
    const up = createBox(0.18, 0.32, 0.18, sleeveCol);
    up.position.set(0, -0.16, 0);
    ag.add(up);
    const lw = createBox(0.16, 0.26, 0.16, sleeveCol);
    lw.position.set(0, -0.40, 0);
    ag.add(lw);
    group.add(ag);
    return ag;
}

// Player Leg Builder
function addLeg(group, xPos, pantsCol) {
    const lg = new THREE.Group();
    lg.position.set(xPos, 0.60, 0);
    const m = createBox(0.23, 0.68, 0.24, pantsCol);
    m.position.set(0, -0.24, 0);
    lg.add(m);
    group.add(lg);
    return lg;
}

// Player Shoe Builder
function addShoe(group, xPos, baseCol, accentCol, soleCol) {
    const sg = new THREE.Group();
    const base = createBox(0.25, 0.12, 0.34, baseCol);
    sg.add(base);
    if (accentCol) {
        const s = createBox(0.27, 0.05, 0.35, accentCol);
        s.position.y = 0.035;
        sg.add(s);
    }
    const sl = createBox(0.26, 0.04, 0.35, soleCol || 0x222222);
    sl.position.y = -0.08;
    sg.add(sl);
    sg.position.set(xPos, 0.07, 0.05);
    group.add(sg);
    return sg;
}

// Player Face Builder
function addFace(group, skinCol, eyeY) {
    [-0.08, 0.08].forEach(ex => {
        const ew = new THREE.Mesh(new THREE.BoxGeometry(0.10, 0.08, 0.04), new THREE.MeshStandardMaterial({ color: 0xffffff }));
        ew.position.set(ex, eyeY, 0.21);
        group.add(ew);
        const ep = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.05, 0.04), new THREE.MeshStandardMaterial({ color: 0x111111 }));
        ep.position.set(ex, eyeY, 0.23);
        group.add(ep);
    });
}

// JAKE BUILDER
function buildJake() {
    const group = new THREE.Group();
    group.rotation.y = Math.PI; // Face away from camera
    const SCALE = 0.95; // Make character smaller

    const SK = 0xfce1c5; // Skin color
    const HO = 0xffffff; // Hoodie (white)
    const VS = 0x1a3a6a; // Vest (blue)
    const SH = 0xcc2222; // Shirt / Cap details (red)
    const JN = 0x1a3a6a; // Jeans / Pants (blue)
    const torso = createBox(0.56 * SCALE, 0.62 * SCALE, 0.36 * SCALE, VS);
    torso.position.set(0, 0.87 * SCALE, 0);
    group.add(torso);

    // Blue Vest Panels (over white hoodie)
    const vf = createBox(0.50 * SCALE, 0.55 * SCALE, 0.06 * SCALE, VS);
    vf.position.set(0, 0.90 * SCALE, 0.19 * SCALE);
    group.add(vf);

    const vl = createBox(0.07 * SCALE, 0.55 * SCALE, 0.32 * SCALE, VS);
    vl.position.set(-0.2 * SCALE, 0.90 * SCALE, 0);
    group.add(vl);

    const vr = createBox(0.07 * SCALE, 0.55 * SCALE, 0.32 * SCALE, VS);
    vr.position.set(0.2 * SCALE, 0.90 * SCALE, 0);
    group.add(vr);

    // addFace(group, SK, 1.48 * SCALE);

    // SUBWAY SURFERS HAT & HOOD
    // White Hoodie/Cap Base
    const hatBase = createBox(0.46 * SCALE, 0.42 * SCALE, 0.44 * SCALE, HO);
    hatBase.position.set(0, 1.55 * SCALE, -0.04 * SCALE);
    group.add(hatBase);

    // Red Front Panel of Cap
    const capFront = createBox(0.47 * SCALE, 0.18 * SCALE, 0.10 * SCALE, SH);
    capFront.position.set(0, 1.66 * SCALE, 0.18 * SCALE);
    group.add(capFront);

    // Red Brim/Visor
    const brim = createBox(0.46 * SCALE, 0.06 * SCALE, 0.34 * SCALE, SH);
    brim.position.set(0, 1.55 * SCALE, 0.34 * SCALE);
    group.add(brim);

    const laG = addArm(group, -0.37 * SCALE, HO);
    const raG = addArm(group, 0.37 * SCALE, HO);
    laG.position.y = 1.18 * SCALE;
    raG.position.y = 1.18 * SCALE;
    const llG = addLeg(group, -0.14 * SCALE, JN);
    const rlG = addLeg(group, 0.14 * SCALE, JN);
    const lShoeG = addShoe(llG, 0, 0x228822, null, 0x222222);
    const rShoeG = addShoe(rlG, 0, 0x228822, null, 0x222222);
    lShoeG.position.set(0, -0.475 * SCALE, 0.05 * SCALE);
    rShoeG.position.set(0, -0.475 * SCALE, 0.05 * SCALE);

    return { group, torso, laG, raG, llG, rlG, lShoeG, rShoeG };
}

// TRAIN BUILDER
function buildTrain() {
    const group = new THREE.Group();
    const cols = [0xcc2222, 0x2244cc, 0x228833, 0xaaaaaa, 0xcc7700, 0x882288, 0x1a6a8a];
    const color = cols[Math.floor(Math.random() * cols.length)];

    const bm = new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.35 });
    const body = new THREE.Mesh(new THREE.BoxGeometry(1.72, 1.95, 3.9), bm);
    body.position.y = 1.10;
    body.castShadow = true;
    body.receiveShadow = true;
    group.add(body);

    const stripe = createBox(1.74, 0.30, 3.92, 0xffffff, { roughness: 0.5 });
    stripe.position.y = 1.65;
    group.add(stripe);

    const wm = new THREE.MeshStandardMaterial({ color: 0x2a4060, emissive: 0x0d1e35, emissiveIntensity: 0.5, roughness: 0.2 });
    [-1, 1].forEach(sx => {
        for (let i = 0; i < 3; i++) {
            const w = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.44, 0.60), wm);
            w.position.set(sx * 0.89, 1.32, -1.2 + i * 1.2);
            group.add(w);
        }
    });

    const wlm = new THREE.MeshStandardMaterial({ color: 0x252525, metalness: 0.92, roughness: 0.3 });
    [-1.3, 1.3].forEach(wz => {
        const wl = new THREE.Mesh(new THREE.CylinderGeometry(0.26, 0.26, 1.88, 12), wlm);
        wl.rotation.z = Math.PI / 2;
        wl.position.set(0, 0.26, wz);
        wl.castShadow = true;
        group.add(wl);
    });

    const fc = new THREE.Color(color);
    fc.multiplyScalar(1.12);
    const front = createBox(1.72, 1.95, 0.14, fc.getHex(), { roughness: 0.5 });
    front.position.set(0, 1.10, 2.0);
    group.add(front);

    const hlm = new THREE.MeshStandardMaterial({ color: 0xffffcc, emissive: 0xffee66, emissiveIntensity: 1.8, roughness: 0.1 });
    [-0.52, 0.52].forEach(hx => {
        const hl = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.15, 0.08), hlm);
        hl.position.set(hx, 0.88, 2.06);
        group.add(hl);
    });

    const ws = createBox(1.0, 0.55, 0.07, 0x3a6080, { emissive: 0x1a3050, emissiveIntensity: 0.3 });
    ws.position.set(0, 1.55, 2.08);
    group.add(ws);

    return group;
}

// LOW BARRIER BUILDER
function buildLowBarrier() {
    const group = new THREE.Group();
    const pm = new THREE.MeshStandardMaterial({ color: 0xdddddd, metalness: 0.55, roughness: 0.5 });
    const bm = new THREE.MeshStandardMaterial({ color: 0xee2222, metalness: 0.3, roughness: 0.6 });
    const sm = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.7 });

    [-0.6, 0.6].forEach(px => {
        const p = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.065, 0.72, 8), pm);
        p.position.set(px, 0.36, 0);
        p.castShadow = true;
        group.add(p);
        const b = createBox(0.22, 0.06, 0.22, 0xdddddd, { metalness: 0.55 });
        b.position.set(px, 0.03, 0);
        group.add(b);
    });

    const bar = createBox(1.30, 0.13, 0.13, 0xee2222, { metalness: 0.3, roughness: 0.6 });
    bar.position.set(0, 0.72, 0);
    group.add(bar);

    for (let bx2 = -0.44; bx2 <= 0.44; bx2 += 0.22) {
        const s = createBox(0.09, 0.13, 0.14, 0xffffff);
        s.position.set(bx2, 0.72, 0);
        group.add(s);
    }
    const bar2 = createBox(1.30, 0.08, 0.08, 0xee2222, { metalness: 0.3 });
    bar2.position.set(0, 0.42, 0);
    group.add(bar2);

    return group;
}

// HIGH FENCE BUILDER
function buildHighFence() {
    const group = new THREE.Group();
    const fm = new THREE.MeshStandardMaterial({ color: 0x888860, metalness: 0.4, roughness: 0.7 });
    const pm = new THREE.MeshStandardMaterial({ color: 0x8B5A2B, roughness: 0.92 });
    const pm2 = new THREE.MeshStandardMaterial({ color: 0x7a4e24, roughness: 0.93 });

    // Side support posts
    [-0.62, 0.62].forEach(px => {
        // Main tall post
        const p = createBox(0.09, 2.5, 0.09, 0x888860, { metalness: 0.4 });
        p.position.set(px, 1.25, 0);
        group.add(p);
        
        // Spike on top
        const sp = new THREE.Mesh(new THREE.ConeGeometry(0.07, 0.18, 4), fm);
        sp.position.set(px, 2.59, 0);
        group.add(sp);
    });

    // Horizontal bars (raised)
    [0.85, 2.35].forEach(py => {
        const b = createBox(1.28, 0.08, 0.08, 0x888860, { metalness: 0.4 });
        b.position.set(0, py, 0);
        group.add(b);
    });

    // Vertical planks (raised)
    for (let i = 0; i < 8; i++) {
        const px = -0.56 + i * (1.12 / 7);
        const pl = createBox(0.13, 1.50, 0.07, i % 2 === 0 ? 0x8B5A2B : 0x7a4e24);
        pl.position.set(px, 1.60, 0);
        group.add(pl);
    }

    // Top yellow trim (raised)
    const tp = createBox(1.28, 0.07, 0.05, 0xffcc00, { roughness: 0.8 });
    tp.position.set(0, 2.40, 0.04);
    group.add(tp);

    return group;
}

// COIN BUILDER
function buildCoin() {
    const coinGeo = new THREE.CylinderGeometry(0.21, 0.21, 0.09, 14);
    const coinMat = new THREE.MeshStandardMaterial({
        color: 0xFFFF00,
        metalness: 0.9,
        roughness: 0.05,
        emissive: 0xFFFF00,
        emissiveIntensity: 1.0
    });
    const mesh = new THREE.Mesh(coinGeo, coinMat);
    mesh.rotation.x = Math.PI / 2;
    return mesh;
}
