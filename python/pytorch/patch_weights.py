import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from ppo import ActorCritic

WEIGHTS_FILE = "python/pytorch/results/weights/score_681_20260502_103807.pth"

# PPO hyperparams (must match Agent in main.py)
LR_ACTOR  = 0.0003
LR_CRITIC = 0.001
EPS_CLIP  = 0.2
C1        = 0.5
C2        = 0.05
GRAD_CLIP = 0.5

# Synthetic pass settings
N_PASSES   = 300   # gradient passes to build stable EMA stats
BATCH_SIZE = 256
SIMULATED_STEP = 3000   # override step so bias-corrections are ≈ 1.0

device = torch.device("cpu")

# ── Load original weights ──────────────────────────────────────────────────────
original_state = torch.load(WEIGHTS_FILE, map_location=device, weights_only=True)

model = ActorCritic(state_dim=16, action_dim=5, layers=[64, 64]).to(device)
model.load_state_dict(original_state)

optimizer = torch.optim.Adam([
    {"params": model.actor.parameters(),  "lr": LR_ACTOR},
    {"params": model.critic.parameters(), "lr": LR_CRITIC},
])

mse_loss = nn.MSELoss()


def random_game_states(n):
    """Generate plausible random game states (16-dim, matching extract_state)."""
    lane      = (torch.randint(0, 3, (n,)).float() - 1.0).unsqueeze(1)   # -1, 0, 1
    y         = (torch.rand(n)).unsqueeze(1)                              # y/3.0
    rolling   = torch.randint(0, 2, (n,)).float().unsqueeze(1)
    speed     = (torch.rand(n) * 0.5 + 0.8).unsqueeze(1)                 # 0.8–1.3

    parts = [lane, y, rolling, speed]

    for _ in range(3):   # obstacle features per lane
        has_obs = (torch.rand(n) > 0.4).float()
        z_norm  = (has_obs * torch.rand(n) + (1.0 - has_obs)).unsqueeze(1)
        # type: 0.0=low, 0.5=high, 1.0=train, -1.0=none
        obs_type = (has_obs * (torch.randint(0, 3, (n,)).float() * 0.5)
                    - (1.0 - has_obs)).unsqueeze(1)
        parts += [z_norm, obs_type]

    for _ in range(3):   # coin features per lane
        has_coin  = (torch.rand(n) > 0.5).float()
        coin_z    = (has_coin * torch.rand(n) + (1.0 - has_coin)).unsqueeze(1)
        coin_cnt  = (has_coin * torch.randint(0, 6, (n,)).float()).unsqueeze(1)
        parts += [coin_z, coin_cnt]

    return torch.cat(parts, dim=1)   # (n, 16)


# ── Synthetic gradient passes ──────────────────────────────────────────────────
print(f"Running {N_PASSES} synthetic PPO-like gradient passes …")

for step in range(N_PASSES):
    states   = random_game_states(BATCH_SIZE)
    actions  = torch.randint(0, 5, (BATCH_SIZE,))
    # Normalized advantages ~ N(0,1), returns in a realistic range
    advantages = torch.randn(BATCH_SIZE)
    returns    = torch.rand(BATCH_SIZE) * 80.0 + 10.0   # 10–90 m

    logprobs, state_values, entropy = model.evaluate(states, actions)
    state_values = state_values.squeeze()
    if state_values.dim() == 0:
        state_values = state_values.unsqueeze(0)

    # Mimic a stable policy: old_logprobs close to current (small KL)
    old_logprobs = (logprobs + torch.randn_like(logprobs) * 0.05).detach()
    ratios = torch.exp(logprobs - old_logprobs)
    surr1  = ratios * advantages
    surr2  = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages

    loss = (-torch.min(surr1, surr2)
            + C1 * mse_loss(state_values, returns)
            - C2 * entropy).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()   # Adam accumulates exp_avg / exp_avg_sq naturally

    if (step + 1) % 50 == 0:
        print(f"  Pass {step + 1}/{N_PASSES}  loss={loss.item():.3f}")

# ── Override step counter → SIMULATED_STEP ────────────────────────────────────
# exp_avg / exp_avg_sq values are realistic from the gradient passes;
# we set step = SIMULATED_STEP so Adam applies bias-correction ≈ 1.0 on resume.
for p in model.parameters():
    if p in optimizer.state:
        s = optimizer.state[p]
        if "step" in s:
            if isinstance(s["step"], torch.Tensor):
                s["step"].fill_(float(SIMULATED_STEP))
            else:
                s["step"] = SIMULATED_STEP

# ── Save new checkpoint ────────────────────────────────────────────────────────
torch.save({
    "policy":    original_state,
    "optimizer": optimizer.state_dict(),
}, WEIGHTS_FILE)
print(f"\nPatched and saved: {WEIGHTS_FILE}")

# ── Quick sanity check ─────────────────────────────────────────────────────────
ckpt = torch.load(WEIGHTS_FILE, map_location=device, weights_only=True)
n_param_states = len(ckpt["optimizer"]["state"])
sample_step    = next(iter(ckpt["optimizer"]["state"].values()))["step"]
print(f"  keys           : {list(ckpt.keys())}")
print(f"  optimizer state: {n_param_states} parameter entries")
print(f"  step value     : {sample_step}")
print("Done.")
