"""Graphes reproductibles : python plot_comparison.py baseline.json challenger.json."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline', type=Path)
    p.add_argument('challenger', type=Path)
    args = p.parse_args()
    reports = [json.loads(path.read_text()) for path in (args.baseline, args.challenger)]
    assert [r['seed'] for r in reports[0]['episodes']] == [r['seed'] for r in reports[1]['episodes']]
    colors = ['#47728b', '#d36537']
    names = ['Cycle hamiltonien', 'BFS torique sécurisé']
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'figure.facecolor':'#f6f8fa', 'axes.facecolor':'#f6f8fa',
                         'axes.labelcolor':'#253747', 'text.color':'#253747'})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    fig.suptitle('Même score. Combien de détours en moins ?', fontsize=23, fontweight='bold', x=.07, ha='left')
    fig.text(.07, .922, '100 parties par méthode • seeds 1000–1099 • grille torique 15 × 15', fontsize=12)
    for report, name, color in zip(reports, names, colors):
        episodes = report['episodes']
        checkpoints = np.arange(1, 224)
        timings = np.array([[r['milestone_steps'].get(str(k), np.nan)/300 for k in checkpoints]
                            for r in episodes])
        mean = np.nanmean(timings, axis=0)
        low, high = np.nanpercentile(timings, [10, 90], axis=0)
        axes[0,0].plot(checkpoints, mean, color=color, label=name, linewidth=2.5)
        axes[0,0].fill_between(checkpoints, low, high, color=color, alpha=.13)
    axes[0,0].set(title='Temps pour atteindre chaque score', xlabel='Score atteint', ylabel='Minutes théoriques à 5 Hz')
    axes[0,0].legend(frameon=False)
    axes[0,0].text(.02,.97,'Bande : 10e–90e percentiles, pas un intervalle de confiance',
                   transform=axes[0,0].transAxes, va='top', fontsize=8)
    durations = [[r['theoretical_seconds_5hz']/60 for r in report['episodes']] for report in reports]
    for i, values in enumerate(durations):
        axes[0,1].scatter(np.full(len(values), i)+np.linspace(-.13,.13,len(values)), values,
                          s=15, alpha=.55, color=colors[i])
        axes[0,1].plot([i-.22,i+.22], [np.mean(values)]*2, color=colors[i], linewidth=4)
    axes[0,1].set(xticks=[0,1], xticklabels=['Cycle', 'BFS sécurisé'], ylabel='Minutes théoriques à 5 Hz',
                  title='Durée finale : chaque point est une partie')
    axes[0,1].set_ylim(bottom=0)
    values = [report['summary']['mean_steps_per_apple'] for report in reports]
    axes[1,0].barh(names, values, color=colors, height=.45)
    for i, value in enumerate(values):
        axes[1,0].text(value+1,i,f'{value:.2f}',va='center',fontweight='bold')
    axes[1,0].set(xlabel='Mouvements par pomme (moyenne)', title='Efficacité du trajet', xlim=(0,max(values)*1.2))
    ax=axes[1,1]
    ax.axis('off')
    baseline, challenger = [report['summary'] for report in reports]
    gain=100*(1-challenger['mean_steps']/baseline['mean_steps'])
    ax.text(0,.92,f'{gain:.1f} % de mouvements en moins',fontsize=20,fontweight='bold')
    for i, (report,name,color) in enumerate(zip(reports,names,colors)):
        s=report['summary']
        ax.text(0,.73-i*.30,name,color=color,fontsize=14,fontweight='bold',va='top')
        ax.text(0,.62-i*.30,f"Score total : {s['score_total']:,}  |  Victoires : {s['victories']}/100\n"
                f"Collisions : {s['collisions']}  |  Tronquées : {s['truncated']}",fontsize=12,va='top')
    ax.text(0,.02,'Simulation ≠ chronométrage d’une partie réelle.\nMême seed ≠ mêmes pommes après divergence des corps.', fontsize=10)
    for ax in axes.flat[:3]:
        ax.grid(axis='y', alpha=.16)
        ax.set_axisbelow(True)
    fig.text(.07,.025,'Les temps sont calculés : mouvements ÷ 5. Aucun changement du moteur ou du rendu du jeu.',fontsize=10)
    fig.tight_layout(rect=[.035,.06,.98,.9],h_pad=3,w_pad=3)
    folder = args.challenger.parent
    fig.savefig(folder/'comparison.png',dpi=170)
    fig.savefig(folder/'comparison.svg')
    print(folder/'comparison.png')


if __name__ == '__main__':
    main()
