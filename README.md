Play game : 

Open HTML file in your browser:
file:///Users/arthurmathorel/Documents/Github/SubwayAI/game/index.html

Train/use AI:
python3 main.py

PascalCase is used for files that export a Class or a React Component, while lowercase/camelCase is used for utility files, constants, or entry points.



## 1. Paramètres du Joueur 
- **Lane** : Position horizontale actuelle (-1.0: gauche, 0.0: milieu, 1.0: droite).
- **Y** : Hauteur actuelle du joueur (normalisée par 3.0).
- **Sliding** : Indique si le joueur est en train de glisser/rouler (0.0 ou 1.0).
- **Speed** : Vitesse actuelle du jeu (normalisée par 10.0).

## 2. Obstacles 
- **LX_Z** : Distance de l'obstacle par rapport au joueur. X varie de 1 à 3. Plus la valeur est proche de 0.0, plus l'obstacle est proche. Défaut à 1.0 si aucun obstacle n'est visible. 
- **LX_T** : Type de l'obstacle pour savoir comment l'esquiver :
    - `-1.0`: **Rien** (Voie libre).
    - `0.0` : **Barrière basse** (on peut sauter par-dessus ou glisser dessous).
    - `0.5` : **Clôture haute** (on doit obligatoirement glisser dessous).
    - `1.0` : **Train** (doit être évité en changeant de voie ou en sautant dessus).

## 3. Pièces (Lanes 1, 2, 3)
Le système suit les pièces sur **chacune des trois voies**, en ne considérant que celles situées **avant** le prochain obstacle de la voie :
- **CX_Z** : Distance de la prochaine pièce (située avant l'obstacle). Défaut à 1.0 si aucune pièce n'est visible avant l'obstacle. Normalisé par 50.0.
- **CX_N** : Nombre de pièces présentes sur cette voie **avant** d'atteindre le prochain obstacle.



To Do : 

IA : 
bug ia reste en train de sauter



Front : 
longueur des sauts de pièces doivent etre plus longs quand vitesse augmente (nb pièces aussi)
bouger un peu la caméra quand on change de line (pas mode IA multiples)