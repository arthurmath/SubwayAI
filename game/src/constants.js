// Game Configuration Constants
const LANES = [-2.2, 0, 2.2];
const BASE_SPEED = 14;
const MAX_SPEED_MULT = 2.5;
const SPEED_STEP = 0.18;
const SCORE_LEVEL_FOR_SPEEDUP = 500;

const JUMP_VELOCITY = 10.5;
const GRAVITY = -28;
const ROLL_DURATION = 0.78;
const LANE_LERP_SPEED = 9.5;
const RECYCLE_Z = 7;

const HIT_X_TRAIN = 0.83;
const HIT_X_BARRIER = 0.73;
const HIT_Z_ENTER = -1.85;
const HIT_Z_EXIT = 1.85;

const AI_PLAYERS = 20;
const AI_ACTION_COOLDOWN_MS = 250; // ms between two AI decisions (~4 actions/sec)

const CAMERA_POS = { x: 0, y: 3.6, z: 5.8 };
const CAMERA_LOOK_AT = { x: 0, y: 0.8, z: -10 };

const COLORS = {
    GRASS: 0x5cb85c,
    SIDEWALK: 0xc8c8c8,
    TRACK_BED: 0x6a6070,
    GRAVEL: 0x7a7060,
    RAIL: 0xcccccc,
    SLEEPER: 0x5a3a18,
    SKY_FOG: 0xb0e0ff,
    SUNLIGHT: 0xfff8e8,
    AMBIENT: 0xd0e8ff,
};
