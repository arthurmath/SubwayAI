/**
 * World class handles the Three.js scene, camera, lights, and static environment.
 */
class World {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.setupRenderer();
        this.setupScene();
        this.setupCamera();
        this.setupLights();
        this.setupEnvironment();

        window.addEventListener('resize', () => this.onResize());
    }

    setupRenderer() {
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;
        this.renderer.setClearColor(0x87CEEB, 0);
        this.container.appendChild(this.renderer.domElement);
    }

    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.Fog(COLORS.SKY_FOG, 28, 95);
    }

    setupCamera() {
        this.camera = new THREE.PerspectiveCamera(62, window.innerWidth / window.innerHeight, 0.1, 200);
        this.camera.position.set(CAMERA_POS.x, CAMERA_POS.y, CAMERA_POS.z);
        this.camera.lookAt(CAMERA_LOOK_AT.x, CAMERA_LOOK_AT.y, CAMERA_LOOK_AT.z);
    }

    setupLights() {
        this.scene.add(new THREE.AmbientLight(COLORS.AMBIENT, 0.72));
        const sun = new THREE.DirectionalLight(COLORS.SUNLIGHT, 1.55);
        sun.position.set(14, 32, 8);
        sun.castShadow = true;
        sun.shadow.camera.near = 1;
        sun.shadow.camera.far = 110;
        sun.shadow.camera.left = -25;
        sun.shadow.camera.right = 25;
        sun.shadow.camera.top = 25;
        sun.shadow.camera.bottom = -25;
        sun.shadow.mapSize.set(1024, 1024);
        sun.shadow.bias = -0.001;
        this.scene.add(sun);
        this.scene.add(new THREE.HemisphereLight(0x87ceeb, 0x5a9e5a, 0.55));
    }

    setupEnvironment() {
        const grassMat = new THREE.MeshStandardMaterial({ color: COLORS.GRASS, roughness: 1 });
        const swMat = new THREE.MeshStandardMaterial({ color: COLORS.SIDEWALK, roughness: 0.95 });
        const tbMat = new THREE.MeshStandardMaterial({ color: COLORS.TRACK_BED, roughness: 0.95 });
        const grvMat = new THREE.MeshStandardMaterial({ color: COLORS.GRAVEL, roughness: 1 });
        const railMat = new THREE.MeshStandardMaterial({ color: COLORS.RAIL, metalness: 0.85, roughness: 0.2 });

        // Tracks area
        const depth = 280;
        [-34, 34].forEach(x => {
            const g = new THREE.Mesh(new THREE.PlaneGeometry(55, depth), grassMat);
            g.rotation.x = -Math.PI / 2;
            g.position.set(x, -0.02, -120);
            g.receiveShadow = true;
            this.scene.add(g);
        });

        [-7.5, 7.5].forEach(x => {
            const s = new THREE.Mesh(new THREE.PlaneGeometry(3.5, depth), swMat);
            s.rotation.x = -Math.PI / 2;
            s.position.set(x, -0.01, -120);
            s.receiveShadow = true;
            this.scene.add(s);
        });

        const tb = new THREE.Mesh(new THREE.PlaneGeometry(12.5, depth), tbMat);
        tb.rotation.x = -Math.PI / 2;
        tb.position.set(0, -0.005, -120);
        tb.receiveShadow = true;
        this.scene.add(tb);

        [-2.2, 0, 2.2].forEach(x => {
            const g = new THREE.Mesh(new THREE.PlaneGeometry(1.2, depth), grvMat);
            g.rotation.x = -Math.PI / 2;
            g.position.set(x, 0.002, -120);
            this.scene.add(g);
        });

        [-2.75, -1.65, -0.55, 0.55, 1.65, 2.75].forEach(x => {
            const r = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.08, depth), railMat);
            r.position.set(x, 0.05, -120);
            r.receiveShadow = true;
            this.scene.add(r);
        });

        // Sleepers (the wood parts)
        this.sleepers = [];
        const SLP_COUNT = 48, SLP_SPACING = 1.2;
        const slpMat = new THREE.MeshStandardMaterial({ color: COLORS.SLEEPER, roughness: 0.96 });
        const slpGeo = new THREE.BoxGeometry(8.4, 0.11, 0.40);
        for (let i = 0; i < SLP_COUNT; i++) {
            const m = new THREE.Mesh(slpGeo, slpMat);
            m.position.set(0, 0.01, -i * SLP_SPACING);
            m.receiveShadow = true;
            m.castShadow = true;
            this.scene.add(m);
            this.sleepers.push({ mesh: m, initialZ: m.position.z, spacing: SLP_SPACING, count: SLP_COUNT });
        }

        // Trees
        this.trees = [];
        const TREE_COUNT = 20, TREE_SPACING = 6.5;
        for (let i = 0; i < TREE_COUNT; i++) {
            const tl = this.makeTree(-1);
            tl.position.z = -i * TREE_SPACING;
            this.trees.push({ group: tl, initialZ: tl.position.z, spacing: TREE_SPACING, count: TREE_COUNT });

            const tr = this.makeTree(1);
            tr.position.z = -i * TREE_SPACING;
            this.trees.push({ group: tr, initialZ: tr.position.z, spacing: TREE_SPACING, count: TREE_COUNT });
        }
    }

    makeTree(side) {
        const group = new THREE.Group();
        const sc = 0.65 + Math.random() * 0.7;
        const pine = Math.random() > 0.45;

        if (pine) {
            const th = 0.5 * sc;
            const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.10 * sc, 0.14 * sc, th, 7), new THREE.MeshStandardMaterial({ color: 0x6b3a1f, roughness: 0.95 }));
            trunk.position.y = th / 2;
            trunk.castShadow = true;
            group.add(trunk);
            const ch1 = 1.6 * sc;
            const c1 = new THREE.Mesh(new THREE.ConeGeometry(0.7 * sc, ch1, 7), new THREE.MeshStandardMaterial({ color: 0x2d6e35, roughness: 0.9 }));
            c1.position.y = th + ch1 / 2;
            c1.castShadow = true;
            group.add(c1);
        } else {
            const th = 0.7 * sc;
            const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.11 * sc, 0.16 * sc, th, 6), new THREE.MeshStandardMaterial({ color: 0x7a4a22, roughness: 0.95 }));
            trunk.position.y = th / 2;
            trunk.castShadow = true;
            group.add(trunk);
            const lr = 0.75 * sc;
            const lc = [0x3a8c3a, 0x4a9e40, 0x2e7a30][Math.floor(Math.random() * 3)];
            const lv = new THREE.Mesh(new THREE.SphereGeometry(lr, 7, 6), new THREE.MeshStandardMaterial({ color: lc, roughness: 0.95 }));
            lv.position.y = th + lr * 0.85;
            lv.castShadow = true;
            group.add(lv);
        }
        group.position.x = side * (8.5 + Math.random() * 6);
        this.scene.add(group);
        return group;
    }

    update(dt, speedMultiplier) {
        const speed = BASE_SPEED * speedMultiplier;
        const ts = speed * 0.32; // Trees move slower

        // Recycle sleepers
        for (const s of this.sleepers) {
            s.mesh.position.z += speed * dt;
            if (s.mesh.position.z > RECYCLE_Z) {
                s.mesh.position.z -= s.count * s.spacing;
            }
        }

        // Recycle trees
        for (const t of this.trees) {
            t.group.position.z += ts * dt;
            if (t.group.position.z > 20) {
                t.group.position.z -= t.count * t.spacing;
            }
        }
    }

    render() {
        this.renderer.render(this.scene, this.camera);
    }

    onResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
}
