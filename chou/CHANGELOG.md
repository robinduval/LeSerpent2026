# Changelog

## 2026-09-16 — Première implémentation RL de cette session

- Préservation du jeu initial contrôlé au clavier et de sa licence dans `baseline/`.
- Audit du tore 15 × 15, des 5 Hz, du score, de la croissance différée et du maximum exact à 223.
- Extraction du moteur avec comparaison transition par transition à la source sur 1 000 graines.
- Implémentation Double DQN, réseaux séparés, replay propriétaire, états 11 et 248 informations, PER et n-step facultatifs.
- Autorisation utilisateur d'accélérer l'entraînement et de modifier ses récompenses ; maintien des règles et de la cadence d'évaluation.
- Recréation du venv par uv sous Python 3.13 ; vérification de pygame 2.6.1 et pygame.font.
- Entraînements bornés, journaux, checkpoints CPU, reprise, évaluations à 5 Hz et classement score puis temps.
- Tests des collisions, de la queue, de la croissance, des cibles, du replay, des reprises, du chronométrage et du lancement.
- Interface autonome sans apprentissage au lancement et erreurs explicites pour les poids manquants.
- Résultats et sélection consignés dans les données d'expérience, sans historique antérieur inventé.

## 2026-09-16 — sprint supplémentaire de vingt minutes

- Ajout d'un réseau Double DQN à valeurs d'action et d'un filtre hamiltonien réorganisable, avec prise en compte de la croissance différée et preuve de progression.
- Expériences distinctes : filtre local tail2, distillation d'entraînement, cycle à rang fixe, cycle dynamique, optimisation d'arcs libres. Les variantes où le réseau intervient trop peu restent expérimentales.
- Trois graines d'entraînement indépendantes pour le cycle dynamique ; 125, 24 et 24 épisodes complets à 223 en apprentissage. Ce ne sont pas des taux d'évaluation à 5 Hz.
- Comparaisons appariées RL pur à 5 Hz : la nouvelle distillation n'améliore pas le modèle initial. Conservation du défaut pur tant que l'utilisation d'un filtre en évaluation reste à confirmer.
- Chargement explicite des familles de checkpoints, interface « RL + filtre », journaux de décisions et coûts de démonstration, archivage du modèle initial et des rapports.
- 55 tests techniques réussis, import pygame.font sous Python 3.13.15, interface native Cocoa et commande sans argument vérifiées.
