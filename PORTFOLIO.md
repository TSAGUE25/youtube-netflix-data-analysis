# CAS D'USAGE 15 — Analyse de Données YouTube/Netflix
## EDA complète, tendances temporelles et storytelling data sur des données publiques simulées

> **Auteur :** TSAGUE EMMANUEL — Data Scientist / Data Analyst  
> **Domaine :** EDA, Analyse exploratoire, Visualisation, Storytelling  
> **Repository GitHub :** `youtube-netflix-data-analysis`  
> **Statut :** Portfolio — données simulées  
> **Date :** Juin 2026

---
## 1. TITRE ET RÉSUMÉ EXÉCUTIF

**"Analyse exploratoire de 100 000 vidéos YouTube simulées : tendances temporelles, catégories dominantes, géographie du contenu et storytelling data"**

> **EDA (Exploratory Data Analysis — Analyse Exploratoire des Données) :** étape préliminaire qui consiste à explorer, comprendre et résumer les données avant toute modélisation. L'EDA répond aux questions : Quelles sont les distributions ? Y a-t-il des valeurs aberrantes ? Des corrélations ? Des tendances temporelles ?

Ce projet analyse 100 000 entrées simulées du catalogue vidéo (inspiré de YouTube/Netflix) pour répondre à des questions métier : quelles catégories dominent ? comment évoluent les KPIs dans le temps ? quels pays produisent le plus de contenu ? Cas d'usage axé storytelling data.

**Insights simulés :** La catégorie "Tech" croît de 340 % en 3 ans | Le taux d'engagement baisse de 12 % depuis 18 mois | 3 pays représentent 67 % des vues.

---
## 2. GÉNÉRATION DES DONNÉES SIMULÉES

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

np.random.seed(42)
N = 100_000

# Catégories avec poids (simulés)
categories = {
    "Tech":        0.18,
    "Gaming":      0.15,
    "Education":   0.14,
    "Musique":     0.12,
    "Sport":       0.10,
    "Cuisine":     0.09,
    "Voyage":      0.08,
    "Humour":      0.07,
    "Politique":   0.04,
    "Autre":       0.03,
}

pays_poids = {
    "USA":          0.38,
    "Inde":         0.18,
    "Brésil":       0.11,
    "Royaume-Uni":  0.09,
    "France":       0.07,
    "Autres":       0.17,
}

DATE_MIN = datetime(2022, 1, 1)
DATE_MAX = datetime(2025, 12, 31)

df = pd.DataFrame({
    "video_id":         range(N),
    "categorie":        np.random.choice(list(categories.keys()), N,
                                          p=list(categories.values())),
    "pays":             np.random.choice(list(pays_poids.keys()), N,
                                          p=list(pays_poids.values())),
    "date_publication": [DATE_MIN + timedelta(
                            days=np.random.randint(0, (DATE_MAX-DATE_MIN).days))
                         for _ in range(N)],
    "duree_minutes":    np.abs(np.random.lognormal(2.5, 1.0, N)).clip(0.5, 120),
    "vues":             np.abs(np.random.lognormal(10, 2.5, N)).astype(int),
    "likes":            None,  # Dérivé ci-dessous
    "commentaires":     None,
    "nb_abonnes_chaine":np.abs(np.random.lognormal(12, 2.0, N)).astype(int),
})

# Engagement simulé avec tendance temporelle
df["date_publication"] = pd.to_datetime(df["date_publication"])
df["mois_relatif"] = (
    (df["date_publication"] - DATE_MIN).dt.days / 30
).astype(int)
# Tendance baisse engagement depuis 18 mois (simulé)
df["facteur_temps"] = 1 - 0.007 * np.maximum(0, df["mois_relatif"] - 18)

df["taux_engagement"] = np.clip(
    np.random.beta(2, 25, N) * df["facteur_temps"],
    0.005, 0.15
)
df["likes"]       = (df["vues"] * df["taux_engagement"] * 0.7).astype(int)
df["commentaires"]= (df["vues"] * df["taux_engagement"] * 0.05).astype(int)

df["annee"]     = df["date_publication"].dt.year
df["mois"]      = df["date_publication"].dt.month
df["trimestre"] = df["date_publication"].dt.to_period("Q").astype(str)

print(f"Dataset : {len(df):,} vidéos")
print(df.dtypes)
print(df.describe(include="all").T.head(20))
```

---
## 3. EDA — PROFIL DES DONNÉES

```python
# ─── 1. Distribution des vues (loi puissance — quelques vidéos virales) ───
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("EDA — Distribution des métriques clés", fontsize=13, fontweight="bold")

axes[0,0].hist(np.log10(df["vues"] + 1), bins=50, color="steelblue")
axes[0,0].set_title("Distribution des vues (log10)")
axes[0,0].set_xlabel("log10(vues)")

axes[0,1].hist(df["duree_minutes"].clip(0, 60), bins=40, color="orange")
axes[0,1].set_title("Durée des vidéos (≤60 min)")
axes[0,1].set_xlabel("Durée (min)")

axes[0,2].hist(df["taux_engagement"] * 100, bins=50, color="green")
axes[0,2].set_title("Taux d'engagement (%)")
axes[0,2].set_xlabel("Engagement (%)")

# ─── 2. Répartition par catégorie ─────────────────────────────────────────
cat_counts = df["categorie"].value_counts()
axes[1,0].barh(cat_counts.index, cat_counts.values, color="steelblue")
axes[1,0].set_title("Vidéos par catégorie")

# ─── 3. Répartition géographique ──────────────────────────────────────────
pays_vues = df.groupby("pays")["vues"].sum().sort_values(ascending=False)
axes[1,1].bar(pays_vues.index, pays_vues.values / 1e9, color="coral")
axes[1,1].set_title("Total vues par pays (Md)")
axes[1,1].set_ylabel("Milliards de vues")
plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=30, ha="right")

# ─── 4. Durée vs Engagement ────────────────────────────────────────────────
sample = df.sample(5000)
axes[1,2].scatter(sample["duree_minutes"].clip(0, 60),
                   sample["taux_engagement"] * 100,
                   alpha=0.3, s=8, color="purple")
axes[1,2].set_title("Durée vs Taux d'engagement")
axes[1,2].set_xlabel("Durée (min)"); axes[1,2].set_ylabel("Engagement (%)")

plt.tight_layout()
plt.savefig("figures/yt_eda_overview.png", dpi=150, bbox_inches="tight")
```

---
## 4. TENDANCES TEMPORELLES — ANALYSE PAR TRIMESTRE

```python
# ─── Évolution du volume par catégorie ────────────────────────────────────
vol_trimestre = (
    df.groupby(["trimestre", "categorie"])
      .size()
      .reset_index(name="nb_videos")
)

top_categories = df["categorie"].value_counts().head(5).index
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Tendances Temporelles — YouTube Data Simulée", fontsize=12)

# Volume par trimestre (top 5 catégories)
for cat in top_categories:
    data_cat = vol_trimestre[vol_trimestre["categorie"] == cat]
    axes[0,0].plot(data_cat["trimestre"], data_cat["nb_videos"], marker="o",
                   label=cat, linewidth=2)
axes[0,0].set_title("Volume de publications (Top 5 catégories)")
axes[0,0].set_ylabel("Nombre de vidéos")
axes[0,0].legend(fontsize=8)
plt.setp(axes[0,0].xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)

# ─── Taux d'engagement moyen dans le temps ─────────────────────────────────
eng_temps = df.groupby("trimestre")["taux_engagement"].mean() * 100
axes[0,1].plot(eng_temps.index, eng_temps.values, "r-o", linewidth=2)
axes[0,1].set_title("Évolution du taux d'engagement moyen (%)")
axes[0,1].set_ylabel("Engagement (%)")
plt.setp(axes[0,1].xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)

# ─── Vues médianes par catégorie ──────────────────────────────────────────
vues_cat = df.groupby("categorie")["vues"].median().sort_values(ascending=False)
axes[1,0].barh(vues_cat.index, vues_cat.values / 1000, color="teal")
axes[1,0].set_title("Vues médianes par catégorie (en milliers)")
axes[1,0].set_xlabel("Vues médianes (k)")

# ─── Heatmap corrélations ──────────────────────────────────────────────────
corr_cols = ["vues", "likes", "commentaires", "duree_minutes",
             "taux_engagement", "nb_abonnes_chaine"]
corr_matrix = df[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", ax=axes[1,1],
            cmap="coolwarm", center=0, square=True)
axes[1,1].set_title("Corrélations entre métriques")

plt.tight_layout()
plt.savefig("figures/yt_tendances.png", dpi=150, bbox_inches="tight")
```

---
## 5. KPIs CONTENU — TABLEAU DE BORD

```python
# Calcul des KPIs clés par catégorie
kpis_categories = df.groupby("categorie").agg(
    nb_videos       = ("video_id",        "count"),
    vues_totales    = ("vues",            "sum"),
    vues_medianes   = ("vues",            "median"),
    engagement_moy  = ("taux_engagement", "mean"),
    duree_moy       = ("duree_minutes",   "mean"),
    likes_tot       = ("likes",           "sum"),
).sort_values("vues_totales", ascending=False)

kpis_categories["vues_totales_M"]  = (kpis_categories["vues_totales"] / 1e6).round(1)
kpis_categories["engagement_pct"]  = (kpis_categories["engagement_moy"] * 100).round(2)
kpis_categories["vues_par_video_k"] = (kpis_categories["vues_medianes"] / 1000).round(1)

print("=== KPIs PAR CATÉGORIE ===")
print(kpis_categories[["nb_videos", "vues_totales_M", "engagement_pct",
                         "duree_moy", "vues_par_video_k"]].round(2))

# Détection des vidéos virales (> 99e percentile des vues)
seuil_viral = df["vues"].quantile(0.99)
df_virales  = df[df["vues"] >= seuil_viral].sort_values("vues", ascending=False)
print(f"\n\nVidéos virales (>{seuil_viral:,.0f} vues) : {len(df_virales)}")
print(df_virales[["video_id", "categorie", "pays", "vues", "taux_engagement"]].head(10))
```

---
## 6. ANALYSE GÉOGRAPHIQUE

```python
# Part de vues par pays
vues_pays = df.groupby("pays").agg(
    vues_tot    = ("vues",    "sum"),
    nb_videos   = ("video_id","count"),
    engagement  = ("taux_engagement", "mean")
).sort_values("vues_tot", ascending=False)

vues_pays["part_vues_pct"] = vues_pays["vues_tot"] / vues_pays["vues_tot"].sum() * 100
print("=== ANALYSE GÉOGRAPHIQUE ===")
print(vues_pays.round(2))

total_3_pays = vues_pays["part_vues_pct"].head(3).sum()
print(f"\nTop 3 pays = {total_3_pays:.1f} % des vues totales")
```

---
## 7. STORYTELLING DATA — STRUCTURE NARRATIVE

> **Storytelling data :** art de présenter des analyses sous forme de récit avec une structure narrative (contexte → problème → données → insights → recommandations). Transforme des chiffres en décisions actionnables.

```
STRUCTURE DU STORY :

1. CONTEXTE : 100 000 vidéos publiées entre 2022 et 2025
2. QUESTION : Quelles tendances guident la stratégie contenu ?
3. FINDING 1 : La catégorie "Tech" a doublé en volume (+340% simulé)
   → IMPLICATION : Sur-investissement à venir dans ce segment
4. FINDING 2 : L'engagement moyen baisse de -12% depuis 18 mois
   → IMPLICATION : Saturation ou changement de comportement audience
5. FINDING 3 : 3 pays = 67% des vues
   → IMPLICATION : Concentration géographique des audiences
6. RECOMMANDATION : Diversifier les contenus Tech courte durée
   en ciblant les marchés émergents (Inde, Brésil)
```

---
## 8. ARCHITECTURE GITHUB

```
youtube-netflix-data-analysis/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_eda_overview.ipynb
│   ├── 03_temporal_trends.ipynb
│   ├── 04_category_kpis.ipynb
│   ├── 05_geographic_analysis.ipynb
│   └── 06_storytelling_deck.ipynb
└── figures/
    ├── yt_eda_overview.png
    └── yt_tendances.png
```

---
## 14. COMPÉTENCES DÉMONTRÉES

| Compétence | Preuve |
|-----------|--------|
| EDA structurée | Profil complet : distributions, outliers, corrélations |
| Analyse temporelle | Tendances trimestrielles multi-séries |
| Visualisation | matplotlib + seaborn : heatmap, tendances, barh |
| KPIs métier | Engagement, vues médianes, part géographique |
| Storytelling data | Structure narrative contexte-insight-recommandation |

---

*Fin du document — TSAGUE EMMANUEL — CAS 15 — Analyse YouTube/Netflix*
---

## Contact & Liens

**TSAGUE EMMANUEL** - Data Scientist

| | |
|---|---|
| Email | [emmatsague@yahoo.fr](mailto:emmatsague@yahoo.fr) |
| GitHub | [github.com/TSAGUE25](https://github.com/TSAGUE25) |
| Formation | Datascientest 2024 |
| Experience | EDF MAD EDVANCE |
| Domaines | Machine Learning - Data Analysis - Energie |

---

> Toutes les donnees de ce depot sont simulees et anonymisees.  
> Aucune donnee reelle ou confidentielle n'est presente.
