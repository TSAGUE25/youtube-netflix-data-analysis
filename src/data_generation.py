import numpy as np
import pandas as pd
from pathlib import Path


def generate_streaming_data(n_videos=5000, n_shows=500, seed=42):
    rng = np.random.default_rng(seed)

    genres = ['Action', 'Comedie', 'Drame', 'Thriller', 'SF', 'Romance',
              'Animation', 'Documentaire', 'Horreur', 'Biopic']
    plateformes = ['YouTube', 'Netflix', 'Disney+', 'Prime Video']
    pays = ['France', 'USA', 'UK', 'Canada', 'Allemagne', 'Espagne']

    # YouTube-style videos
    videos = pd.DataFrame({
        'id':          range(n_videos),
        'plateforme':  rng.choice(['YouTube'], n_videos),
        'genre':       rng.choice(genres, n_videos),
        'annee':       rng.integers(2015, 2024, n_videos),
        'mois':        rng.integers(1, 13, n_videos),
        'duree_min':   rng.integers(3, 60, n_videos),
        'pays_origine': rng.choice(pays, n_videos, p=[0.30, 0.30, 0.15, 0.10, 0.08, 0.07]),
        'abonnes_chaine': np.clip(rng.lognormal(10, 2, n_videos), 100, 50_000_000).astype(int),
    })
    videos['vues']          = np.clip(rng.lognormal(8, 2.5, n_videos), 100, 200_000_000).astype(int)
    videos['likes']         = np.clip((videos['vues'] * rng.beta(2, 20, n_videos)).round(0), 0, None).astype(int)
    videos['commentaires']  = np.clip((videos['vues'] * rng.beta(1, 100, n_videos)).round(0), 0, None).astype(int)
    videos['taux_retention'] = np.clip(rng.beta(4, 2, n_videos) * 100, 10, 95).round(1)
    videos['score_critique'] = np.clip(rng.normal(6.5, 1.5, n_videos), 1, 10).round(1)
    videos['tendance']      = (videos['vues'] > videos['vues'].quantile(0.90)).astype(int)

    # Netflix/VOD shows
    shows = pd.DataFrame({
        'id':          range(n_shows),
        'plateforme':  rng.choice(['Netflix', 'Disney+', 'Prime Video'], n_shows,
                                  p=[0.50, 0.25, 0.25]),
        'genre':       rng.choice(genres, n_shows),
        'annee':       rng.integers(2010, 2024, n_shows),
        'nb_saisons':  rng.integers(1, 8, n_shows),
        'pays_origine': rng.choice(pays, n_shows, p=[0.15, 0.45, 0.15, 0.10, 0.08, 0.07]),
        'duree_ep_min': rng.integers(20, 90, n_shows),
        'score_critique': np.clip(rng.normal(7.0, 1.2, n_shows), 1, 10).round(1),
        'note_spectateurs': np.clip(rng.normal(6.8, 1.3, n_shows), 1, 10).round(1),
        'nb_abonnes_impact': np.clip(rng.lognormal(12, 1.5, n_shows), 0, None).astype(int),
    })

    return videos, shows


def load_or_generate(csv_path, **kwargs):
    path = Path(csv_path)
    shows_path = path.parent / 'shows_simulated.csv'
    if path.exists() and shows_path.exists():
        return pd.read_csv(path), pd.read_csv(shows_path)
    videos, shows = generate_streaming_data(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    videos.to_csv(path, index=False)
    shows.to_csv(shows_path, index=False)
    return videos, shows
