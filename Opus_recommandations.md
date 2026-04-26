Voici une analyse détaillée des leviers à ta disposition. Je classe les recommandations par **impact attendu** (du plus fort au plus faible). Les premières (1-3) sont les plus importantes — il y a probablement un facteur 2-5x à gagner rien qu'avec ça.

## 1. Bug critique : `c1` est ~50x trop petit ✅

Dans `python/pytorch/main.py:29`, tu utilises `c1=0.01`. Le standard PPO est `c1=0.5`.

```125:130:python/pytorch/ppo.py
ratios = torch.exp(logprobs - old_logprobs.detach())

surr1 = ratios * advantages
surr2 = torch.clamp(ratios, 1-self.eps, 1+self.eps) * advantages

loss = -torch.min(surr1, surr2) + self.c1 * self.loss(state_values, rewards) - self.c2 * dist_entropy
```

Avec `c1=0.01`, le critique (qui estime la valeur de l'état) apprend ~50x plus lentement que la politique. Or les **avantages** dépendent directement du critique : `advantages = rewards - state_values`. Si le critique est mauvais, les avantages sont bruités → gradient policy bruité → apprentissage très lent.

→ **Passe `c1` à 0.5**. C'est sans doute le plus gros levier de ce code.

## 2. Implémenter le GAE (Generalized Advantage Estimation) ✅

Actuellement tu utilises des retours Monte-Carlo bruts :

```100:116:python/pytorch/ppo.py
# Monte Carlo estimate of returns
rewards = []
discounted_reward = 0
for reward, is_terminal in zip(reversed(buffer.rewards), reversed(buffer.is_terminals)):
    if is_terminal:
        discounted_reward = 0
    discounted_reward = reward + (self.gamma * discounted_reward)
    rewards.insert(0, discounted_reward)
    
# Normalize rewards
rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
if len(rewards) > 1:
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

old_states, old_actions, old_logprobs, old_state_values = format(buffer)

advantages = rewards.detach() - old_state_values.detach()
```

C'est *l'option à variance maximale*. Le GAE (avec `λ=0.95`) interpole entre TD(0) et Monte-Carlo, ce qui réduit massivement la variance du gradient → convergence beaucoup plus rapide. Quasi tous les benchmarks PPO l'utilisent.

```python
gae_lambda = 0.95
advantages = []
gae = 0.0
values = old_state_values.tolist() + [0.0]  # bootstrap = 0 si terminal
for i in reversed(range(len(buffer.rewards))):
    next_v = 0.0 if buffer.is_terminals[i] else values[i + 1]
    delta = buffer.rewards[i] + self.gamma * next_v - values[i]
    gae = delta + self.gamma * gae_lambda * (0.0 if buffer.is_terminals[i] else gae)
    advantages.insert(0, gae)
advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
returns = advantages + old_state_values  # cible du critic
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

Au passage : tu **normalises les retours** (`rewards`) puis tu calcules l'avantage. La pratique standard est de **normaliser les avantages**, pas les retours — sinon le critique tente de prédire des cibles dont l'échelle change à chaque batch (instable, ce qui explique probablement pourquoi tu as dû mettre `c1` aussi bas pour compenser).

## 3. Ajouter du gradient clipping ✅

C'est dans le "PPO Implementation Matters" (Ilyas et al.) parmi les détails les plus importants. À ajouter avant `optimizer.step()` :

```python
self.optimizer.zero_grad()
loss.mean().backward()
torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
self.optimizer.step()
```

Ça évite des updates catastrophiques quand un avantage est très grand (ex. mort imminente avec ta pénalité de -50).

## 4. Recalibrer la récompense de mort ✅

```210:212:python/pytorch/main.py
if dead:
    reward -= 50.0
```

Les rewards typiques par pas tournent autour de ±5. Une pénalité de -50 crée des spikes énormes dans le gradient et augmente la variance. Essaye **-5 à -10**. Combiné au gradient clipping ci-dessus, ça stabilisera nettement.

Idem la "danger penalty" de -3 par pas : appliquée *à chaque* frame pendant que l'obstacle est dans la zone -20<z<0, ça peut s'accumuler à plusieurs dizaines d'unités sur un seul approche. Vérifie si c'est répété ou one-shot. X

## 5. Hyperparamètres PPO à ajuster

| Paramètre | Actuel | Recommandé | Pourquoi |
|---|---|---|---|
| `c1` ✅ | 0.01 | **0.5** | cf. point 1 |
| `gamma` ✅| 0.99 | **0.97-0.98** | Récompenses très denses (chaque mètre), horizon long inutile, ajoute de la variance |
| `epochs` ✅ | 10 | **4** | 10 sur des données on-policy provoque sur-apprentissage du batch. PPO original utilise 4 (ou 10 avec minibatches) |
| `eps` | 0.2 | 0.2 | OK, standard |
| `c2` | 0.05 | 0.01 → 0.001 | Décroissance dans le temps. 0.05 force trop d'exploration en fin d'entraînement |
| `lr_actor` | 3e-4 | 3e-4 | OK |
| `lr_critic` | 1e-3 | 1e-3 ou 3e-4 | OK une fois `c1` corrigé |
| `UPDATE_TIMESTEP` ✅ | 1000 | **2048-4096** | Batch plus stable, surtout combiné au GAE |
| `layers` | [64, 64] | [128, 128] | Si tu as un GPU, capacité plus large = apprentissage plus rapide en wall-clock per-step |

Pour la planification de l'entropie, tu peux faire un linear decay :
```python
self.c2 = max(0.001, 0.05 * (1.0 - train_count / 2000))
```

## 6. Mini-batches pendant les epochs ✅

Avec un buffer de 2048-4096 samples, faire 4 epochs **full-batch** est sous-optimal. La pratique standard PPO est de tirer des minibatches de 64-256 samples à chaque epoch. Ça donne plus d'updates par epoch sans biaiser.

```python
batch_size = 256
indices = torch.randperm(len(old_states))
for _ in range(self.epochs):
    for start in range(0, len(old_states), batch_size):
        idx = indices[start:start + batch_size]
        # forward + loss + backward sur idx
```

## 7. Côté collecte de données (JS)

```18:19:game/src/constants.js
const AI_PLAYERS = 20;
const AI_ACTION_COOLDOWN_MS = 250; // ms between two AI decisions (~4 actions/sec)
```

À 250 ms × 14×2.5 m/s en pic = ~9 m parcourus entre deux décisions, c'est très grossier en fin de partie. Si ton agent meurt parce qu'il n'a pas le temps de réagir aux obstacles à haute vitesse, **baisser à 150-200 ms** débloquerait le plafond. Tu peux aussi rendre le cooldown adaptatif (`250 / speedMultiplier`). => Non sinon les agents font trop d'actions chaque secondes et meurent.

Et `AI_PLAYERS=20` est une bonne idée — plus de parallélisme = buffer rempli plus vite. Si la machine suit, monter à 30-40 accélèrerait proportionnellement la collecte.

## 8. Représentation d'état — gain qualitatif

Tu donnes la position des obstacles (`z`) mais **pas leur vitesse relative**. À vitesse fixe c'est OK, mais comme la vitesse du jeu accélère (`SPEED_ACCEL`), un même `z=-20` n'a pas la même urgence à 14 m/s qu'à 35 m/s. Ajouter le `time-to-collision = |z| / speed` comme feature peut accélérer significativement l'apprentissage à haute vitesse.

---

## Plan d'action recommandé

Si tu veux le maximum d'impact pour le minimum de changements, fais dans cet ordre :

1. **`c1=0.5`** dans `main.py` — 1 caractère (probablement 2-3x sur la vitesse de convergence)
2. **GAE + normalisation des avantages** dans `ppo.py` — ~15 lignes
3. **Gradient clipping** + **réduire la pénalité de mort à -10** — 2 lignes
4. **`gamma=0.98`, `epochs=4`, `UPDATE_TIMESTEP=2048`** — paramètres simples




# Effet du passage [64, 64] → [128, 128]

## Capacité du modèle

Avec `state_dim=16` et `action_dim=5`, le nombre de paramètres (par réseau actor/critic) :

| Architecture | Paramètres (par réseau) |
|---|---|
| [64, 64] | 16·64 + 64·64 + 64·5 ≈ **5 500** |
| [128, 128] | 16·128 + 128·128 + 128·5 ≈ **19 200** |

C'est ~3.5× plus de capacité. Comme tu as **deux** réseaux (actor + critic), tu passes d'environ 11k à 38k paramètres au total. C'est encore très petit pour un GPU/CPU moderne.

## Effets attendus

**Positifs :**
- **Meilleur plafond de performance** : ton état (16 features) encode des relations non-linéaires (combinaisons lane × type d'obstacle × distance × vitesse × glissement). [64, 64] est probablement *limite* pour bien représenter toutes les situations à haute vitesse, surtout les triple-obstacles. Tu pourrais voir le score moyen plafonner plus haut.
- **Meilleure modélisation du critic** : un critic plus précis = des avantages moins bruités = gradients policy plus propres → convergence plus rapide en nombre d'itérations.

**Négatifs :**
- **Risque accru d'overfitting au batch on-policy** : avec PPO et seulement 2048 samples par update, un réseau plus large peut sur-apprendre le batch courant. Garde `epochs=4` (et surtout pas 10), et le clip PPO + grad clip que tu as déjà ajoutés atténuent ça.
- **Démarrage légèrement plus lent** : les premières dizaines d'itérations peuvent être un peu plus erratiques le temps que les couches plus larges s'organisent.
- **Coût compute** : ~3.5× les FLOPs par forward/backward. Sur CPU ça peut se sentir (les `train()` plus longs bloquent le serveur websocket pendant que les agents JS attendent). Sur GPU c'est négligeable. Vu tes courbes (~4000 itérations, plusieurs minutes par batch), ça reste largement gérable.

## Recommandation pratique

Pour ton problème spécifique (16 features structurées, 5 actions), **[128, 128] est probablement le sweet spot**. Au-delà ([256, 256]) tu n'observerais probablement plus aucun gain — l'état est trop simple.

Une variante intéressante si tu vois que le critic apprend mal : asymétrie **actor [64, 64] / critic [128, 128]**. Le critic bénéficie souvent plus de la capacité supplémentaire (il prédit une régression continue), et l'actor reste compact (politique simple à 5 actions).

## À surveiller après le changement

Sur les courbes :
1. La **moyenne mobile** doit continuer à monter (sinon overfitting ou instabilité — réduis `lr` à 1e-4 / 3e-4).
2. La **value loss** (que tu peux logger via `self.loss(state_values, returns).item()`) doit décroître plus régulièrement qu'avant.
3. Si l'entropie chute trop vite (l'agent devient déterministe trop tôt), monte temporairement `c2`.

En résumé : changement à faible risque, gain probable mais modeste (peut-être +10 à +30% sur le plafond de score), à combiner avec les autres optimisations que tu as déjà appliquées.