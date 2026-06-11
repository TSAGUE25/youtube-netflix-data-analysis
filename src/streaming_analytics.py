import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def engagement_metrics(videos):
    df = videos.copy()
    df['like_rate']     = df['likes'] / df['vues'].clip(lower=1) * 100
    df['comment_rate']  = df['commentaires'] / df['vues'].clip(lower=1) * 100
    df['engagement']    = df['like_rate'] + df['comment_rate']
    return df


def top_content(videos, n=10, by='vues'):
    return videos.nlargest(n, by)[['genre', 'annee', 'duree_min', by, 'tendance']]


def trend_by_year(videos):
    return (videos.groupby('annee')
            .agg(total_vues=('vues', 'sum'),
                 nb_videos=('id', 'count'),
                 vues_moy=('vues', 'mean'),
                 pct_tendance=('tendance', 'mean'))
            .round(2))


def genre_performance(videos):
    return (videos.groupby('genre')
            .agg(vues_moy=('vues', 'mean'),
                 likes_moy=('likes', 'mean'),
                 nb_videos=('id', 'count'),
                 retention_moy=('taux_retention', 'mean'))
            .sort_values('vues_moy', ascending=False)
            .round(1))


def platform_comparison(shows):
    return (shows.groupby('plateforme')
            .agg(score_critique_moy=('score_critique', 'mean'),
                 note_spec_moy=('note_spectateurs', 'mean'),
                 nb_shows=('id', 'count'),
                 nb_saisons_moy=('nb_saisons', 'mean'))
            .round(2))


def plot_eda(videos, shows):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1. Views distribution (log)
    axes[0, 0].hist(np.log10(videos['vues'].clip(lower=1)), bins=50, color='#FF0000', alpha=0.8)
    axes[0, 0].set_title('Distribution des vues (log10 — YouTube)')
    axes[0, 0].set_xlabel('log10(vues)')

    # 2. Genre views
    gp = genre_performance(videos)
    gp['vues_moy'].plot.bar(ax=axes[0, 1], color='#2196F3', edgecolor='white')
    axes[0, 1].set_title('Vues moyennes par genre'); axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Trend by year
    tp = trend_by_year(videos)
    tp['vues_moy'].plot(ax=axes[0, 2], color='#4CAF50', marker='o')
    axes[0, 2].set_title('Vues moyennes par année'); axes[0, 2].set_xlabel('Année')

    # 4. Like rate vs views
    sample = videos.sample(min(1000, len(videos)), random_state=42)
    axes[1, 0].scatter(np.log10(sample['vues'].clip(1)), sample['like_rate'],
                       alpha=0.4, s=8, color='#FF9800')
    axes[1, 0].set_xlabel('log10(vues)'); axes[1, 0].set_ylabel('Like rate (%)')
    axes[1, 0].set_title('Vues vs Engagement')

    # 5. Netflix/VOD platform comparison
    pp = platform_comparison(shows)
    pp['score_critique_moy'].plot.bar(ax=axes[1, 1], color='#9C27B0', edgecolor='white')
    axes[1, 1].set_title('Score critique moyen — VOD')
    axes[1, 1].tick_params(axis='x', rotation=30)

    # 6. Shows retention by genre
    shows_genre = shows.groupby('genre')['score_critique'].mean().sort_values(ascending=False)
    shows_genre.plot.bar(ax=axes[1, 2], color='#00BCD4', edgecolor='white')
    axes[1, 2].set_title('Score critique — VOD par genre')
    axes[1, 2].tick_params(axis='x', rotation=45)

    plt.suptitle('Analyse Streaming — YouTube & VOD', fontweight='bold', fontsize=13)
    plt.tight_layout(); plt.show()


def plot_heatmap_engagement(videos):
    pivot = (videos.groupby(['genre', 'annee'])['vues']
             .mean().unstack(fill_value=0) / 1e6)
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=45)
    ax.set_yticks(range(len(pivot.index)));  ax.set_yticklabels(pivot.index)
    plt.colorbar(im, ax=ax, label='Vues moyennes (M)')
    ax.set_title('Heatmap vues — Genre × Année', fontweight='bold')
    plt.tight_layout(); plt.show()
