# Analyse de Données YouTube/Netflix

> **EDA complète, tendances temporelles et storytelling data sur 100 000 vidéos simulées**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Domaine](https://img.shields.io/badge/Domaine-EDA-green)
![Statut](https://img.shields.io/badge/Statut-Portfolio-orange)
![Données](https://img.shields.io/badge/Données-Simulées%2FAnonymisées-lightgrey)

---

## Contexte métier

Les plateformes de streaming et de vidéo génèrent des masses de données sur les comportements des utilisateurs. L'analyse de ces données permet d'identifier les tendances, les catégories portantes et les patterns d'engagement.

---

## Problème traité

100 000 entrées vidéo simulées (2022-2025). Identifier les tendances par catégorie, détecter la baisse d'engagement, analyser la répartition géographique et structurer les insights en storytelling data.

---

## Solution proposée

EDA complète : distributions, outliers, corrélations. Analyse temporelle trimestrielle multi-catégories. Heatmap de corrélations. Détection vidéos virales (99e percentile). Storytelling data structuré contexte → insight → action.

---

## Technologies utilisées

| Outil | Usage |
|-------|-------|
| Python 3.10+ | Langage principal |
| pandas / numpy | Manipulation des données |
| scikit-learn | Machine Learning & preprocessing |
| matplotlib / seaborn | Visualisation |
| Jupyter Notebook | Exploration interactive |

> Voir `requirements.txt` pour la liste complète.

---

## Structure du projet

```
youtube-netflix-data-analysis/
├── README.md              ← Ce fichier
├── PORTFOLIO.md           ← Documentation complète du cas d'usage
├── .gitignore
├── requirements.txt
├── notebooks/             ← Jupyter Notebooks d'exploration
├── src/                   ← Code Python modulaire
├── data_sample/           ← Données simulées (anonymisées)
├── figures/               ← Graphiques et visualisations
├── reports/               ← Rapports et synthèses
└── docs/                  ← Documentation complémentaire
```

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/TSAGUE25/youtube-netflix-data-analysis.git
cd youtube-netflix-data-analysis

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate    # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer Jupyter
jupyter notebook
```

---

## Métriques clés (données simulées)

```
Taux d'engagement, vues médianes, part géographique, tendance trimestrielle
```

---

## Valeur métier

Insights actionnables pour la stratégie contenu. Identification des catégories croissantes.

---

## Limites

Données 100 % simulées. Pas d'API YouTube/Netflix réelle.

---

## Prochaines améliorations

Intégration YouTube Data API v3. NLP sur titres/descriptions. Prédiction viralité.

---

## Avertissement — Confidentialité

> **Toutes les données utilisées dans ce projet sont simulées, synthétiques ou anonymisées.**
> Aucune donnée réelle, confidentielle ou propriétaire n'est présente dans ce dépôt.
> Ce projet est un cas d'usage pédagogique à destination du portfolio professionnel d'Emmanuel TSAGUE.

---

## Contributors

**TSAGUE EMMANUEL** - Data Scientist  
Specialise en Machine Learning, Data Analysis et systemes decisionnels.  
Formation Datascientest 2024 | EDF MAD EDVANCE  
Email : [emmatsague@yahoo.fr](mailto:emmatsague@yahoo.fr)  
GitHub : [github.com/TSAGUE25](https://github.com/TSAGUE25)

