"""Produit le bilan et les graphes d'une campagne compare_methods.py."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = dict(hamiltonian='#6e8796', **{'bfs-safe':'#8b719c'},
              shortcut='#2d8d88', lookahead='#d69637', dynamic='#cf623f')


def label(report):
    c = report['config']
    return f"{c['algorithm']} / seuil {c['cutoff']} / géométrie {c['turn_offset']}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    args = p.parse_args()
    folder = args.folder
    screening = json.loads((folder/'screening.json').read_text())
    validation = json.loads((folder/'validation.json').read_text())
    selection = json.loads((folder/'selection.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'#f6f8fa','axes.facecolor':'#f6f8fa',
                         'text.color':'#253747','axes.labelcolor':'#253747'})
    fig = plt.figure(figsize=(17,12))
    grid = fig.add_gridspec(2,2,width_ratios=[1.5,1],hspace=.4,wspace=.5)
    ax = fig.add_subplot(grid[:,0])
    ordered = sorted(screening,key=lambda r:r['summary']['mean_steps_per_apple'])
    for i,r in enumerate(ordered):
        s = r['summary']
        bad = s['collisions']+s['truncated']
        val = s['mean_steps_per_apple']/5
        ax.barh(i,val,color=COLORS[r['config']['algorithm']],height=.65,alpha=.9)
        ax.text(val+.08,i,f"{val:.2f}"+(f"  ✕ {bad} échecs" if bad else ''),va='center',fontsize=8)
    ax.set_yticks(range(len(ordered)), [label(r) for r in ordered],fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0,max(r['summary']['mean_steps_per_apple']/5 for r in ordered)*1.35)
    ax.set(title='Réglage : 100 parties par configuration',xlabel='Secondes théoriques par pomme (5 Hz)')
    ax.grid(axis='x',alpha=.15)
    ax.set_axisbelow(True)
    ax2=fig.add_subplot(grid[0,1])
    for i,r in enumerate(validation):
        with gzip.open(folder/r['archive'],'rt') as handle:
            episodes=json.load(handle)['episodes']
        values=np.array([e['theoretical_seconds_5hz']/60 for e in episodes])
        low,high=np.percentile(values,[10,90]);mean=values.mean()
        ax2.plot([low,high],[i,i],linewidth=6,color=COLORS[r['config']['algorithm']],alpha=.5)
        ax2.scatter([mean],[i],s=65,color=COLORS[r['config']['algorithm']])
        ax2.text(high+.12,i,f'{mean:.2f} min',va='center',fontsize=9)
    ax2.set_yticks(range(len(validation)),[label(r) for r in validation],fontsize=8)
    ax2.invert_yaxis()
    ax2.set_ylim(len(validation)-.5,-.5)
    ax2.set_xlim(left=0,right=max(r['summary']['theoretical_seconds_5hz_max']/60 for r in validation)*1.2)
    ax2.set(title='Validation : 1 000 nouvelles seeds',xlabel='Durée théorique à 5 Hz (minutes)')
    ax2.text(0,-.20,'Point : moyenne · trait : 10e–90e percentiles',transform=ax2.transAxes,fontsize=9)
    ax2.grid(axis='x',alpha=.15)
    ax3=fig.add_subplot(grid[1,1]);ax3.axis('off')
    winner=selection['winner']
    if winner:
        s=winner['summary']
        ax3.text(0,.94,'Version retenue',fontsize=20,fontweight='bold',va='top')
        ax3.text(0,.80,label(winner),fontsize=11,va='top')
        ax3.text(0,.68,f"{s['victories']} / {s['parties']} victoires\n"
                 f"{s['score_total']:,} points cumulés\n"
                 f"{s['mean_steps_per_apple']/5:.2f} s par pomme (théorique)\n"
                 f"{s['collisions']} collisions · {s['truncated']} troncatures",fontsize=14,linespacing=1.9,va='top')
    ax3.text(0,.10,'Cell Tree 2×2 : non testé, pavage incompatible\navec 225 cases sans adaptation supplémentaire.',fontsize=10,va='top')
    fig.suptitle('Sélection du serpent : score d’abord, vitesse ensuite',fontsize=22,fontweight='bold',x=.04,ha='left')
    fig.text(.04,.94,'35 configurations · moteur et rendu inchangés · simulations sans attente de clock',fontsize=12)
    fig.subplots_adjust(left=.23,right=.96,top=.87,bottom=.08)
    fig.text(.04,.025,'Durées calculées : mouvements ÷ 5. Aucun score de simulation n’est présenté comme une partie chronométrée.',fontsize=10)
    fig.savefig(folder/'comparison.png',dpi=150)
    fig.savefig(folder/'comparison.svg')
    lines=['# Résultats de la sélection','',
           'Même moteur torique 15×15 et croissance différée. Durées théoriques à 5 Hz, non mesurées en partie réelle.',
           '', '35 configurations sur 100 seeds (1000–1099), finalistes figés puis validation sur 1000 seeds (10000–10999).',
           '', 'Cell Tree 2×2 non testé : pavage incompatible avec 225 cases ; adaptation supplémentaire nécessaire.',
           '', '## Validation', '', '| Configuration | Score total | Victoires | Collisions | s/pomme théoriques | Durée moyenne (min) |',
           '|---|---:|---:|---:|---:|---:|']
    for r in validation:
        s=r['summary']
        lines.append(f"| {label(r)} | {s['score_total']} | {s['victories']}/{s['parties']} | {s['collisions']} | {s['mean_steps_per_apple']/5:.3f} | {s['theoretical_seconds_5hz_mean']/60:.3f} |")
    lines += ['', '## Tous les essais de réglage', '', '| Configuration | Score total | Victoires | Collisions | s/pomme théoriques |', '|---|---:|---:|---:|---:|']
    for r in screening:
        s=r['summary']
        lines.append(f"| {label(r)} | {s['score_total']} | {s['victories']}/{s['parties']} | {s['collisions']} | {s['mean_steps_per_apple']/5:.3f} |")
    lines += ['', 'Chaque archive `.json.gz` contient tous les scores individuels, seeds, directions, jalons et temps de décision.',
              'Les temps de calcul des lots dépendent de la concurrence CPU ; ils ne sont pas des temps de jeu.',
              'Même seed ne signifie pas mêmes positions de pommes après divergence des corps.',
              'La sélection compare les finalistes sur validation : elle ne constitue pas une estimation indépendante après sélection ni une preuve universelle de sûreté.',
              '', '![Comparaison](comparison.png)', '']
    (folder/'REPORT.md').write_text('\n'.join(lines))
    print(folder/'comparison.png')


if __name__ == '__main__':
    main()
