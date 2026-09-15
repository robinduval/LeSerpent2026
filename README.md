# 🐍 PROJET : JEU DE SERPENT EN PYTHON 🐍
***

## 🌟 Aperçu du Projet 

Le classique jeu du **Serpent** (`Snake`) réimplémenté en Python. Ce dépôt sert de base pour l'implémentation d'algorithmes de recherche de chemin et d'agents d'apprentissage par renforcement (RL).

```text
       * * * * * * * * * * * * * * * * * *
      * S N A K E  P R O J E C T   *
     * * * * * * * * * * * * * * * * * *
    
    
     ╔═══════════════════════════════════╗
     ║ ################################# ║
     ║ # @@@                           # ║
     ║ # @ @        [FOOD]             # ║
     ║ # @@@                           # ║
     ║ #                               # ║
     ║ #       ^ ^      _              # ║
     ║ #      ( o )    ( )             # ║
     ║ #       ---     | |             # ║
     ║ #                               # ║
     ║ #                               # ║
     ║ ################################# ║
     ╚═══════════════════════════════════╝
```

Le dépôt est organisé pour la gestion de 25 groupes de travail.
- **`serpent.py`** : 🎮 Jeu de Base. Contient le jeu de Serpent fonctionnel avec tous les mécanismes et le visuel attendus. C'est le point de départ pour toutes les implémentations.
- **`abricot/` à `tomate/`** : 📂 Répertoires de groupe. Chaque dossier est dédié à un groupe pour le dépôt des livrables et contient :
    - **`XX/README`** : C'est ici que tu mets la timeline
    - **`XX/AUTHORS`** : NOM Prénom login
    - **`XX/snake-algo.py`** : Version du jeu exploitant un algorithme de recherche. (22/09)
    - **`XX/snake-ia.py`** : Version du jeu exploitant l'Apprentissage par Renforcement (16/09).

1. Version Algorithmique Avancée (snake-algo.py) (à faire 22/09)

Ce fichier doit contenir une version du jeu de Serpent pilotée par un algorithme déterministe (ex: A*, Dijkstra, BFS, DFS) visant à optimiser la survie ou le score.
```text
    ┌──────────────────────┐
    │  S N A K E - A L G O │
    ├──────────────────────┤
    │ D J I K S T R A      │
    │ A * (A-STAR)         │
    │ B F S / D F S        │
    └──────────────────────┘
```

2. Version Apprentissage par Renforcement (snake-ia.py) (à faire 16/09)

Ce fichier doit contenir une version du jeu exploitant l'Apprentissage par Renforcement (RL). L'agent doit apprendre à jouer de manière autonome. 

```text
    ╔═══════════════════════════════╗
    ║       █ LEARNING AGENT █      ║
    ║  ( R E I N F O R C E M E N T )║
    ║ ( L E A R N I N G - R L )     ║
    ╚═══════╦═══════════════════════╝
            ║
            ▼
    ╔═══════╩═════════════════════════╗
    ║        _ _ _ _ _ _ _ _ _        ║
    ║       ( Q - L E A R N I N G )   ║
    ║       (   D Q N   ( D E E P ) ) ║
    ║        ¯ ¯ ¯ ¯ ¯ ¯ ¯ ¯ ¯ ¯      ║
    ╚═════════════════════════════════╝
```
