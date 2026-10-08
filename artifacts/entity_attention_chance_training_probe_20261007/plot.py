"""Render standalone diagnostic curves from completed JSON; no model loading."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ART=Path(__file__).resolve().parent
ARMS=('native','chance','cpw')
LABELS={'native':'Determinized','chance':'Unrestricted chance','cpw':'CPW k=.25, alpha=.5'}
COLORS={'native':'#455a64','chance':'#1976d2','cpw':'#d97706'}

if __name__=='__main__':
    analysis=json.loads((ART/'analysis.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for arm in ARMS:
        rows=analysis['training'][arm]['curve'];steps=[r['step'] for r in rows]
        for split,style in (('validation','-'),('train','--')):
            base=rows[0][split]['policy_kl']
            axes[0,0].plot(steps,[r[split]['policy_kl']-base for r in rows],style,color=COLORS[arm],
                           marker='o',label=LABELS[arm] if split=='validation' else None)
        own0=rows[0]['validation']['value_loss']; common0=rows[0]['common']['summary']['all']['value_mse']
        axes[0,1].plot(steps,[r['validation']['value_loss']-own0 for r in rows],'-o',color=COLORS[arm])
        axes[0,1].plot(steps,[r['common']['summary']['all']['value_mse']-common0 for r in rows],'--o',color=COLORS[arm])
        axes[0,2].plot(steps,[r['common']['summary']['all']['policy_kl_to_frozen_mean_targets']['native'] for r in rows],'-o',color=COLORS[arm])
    axes[0,0].set_title('Policy KL change: holdout solid, train dashed')
    axes[0,1].set_title('Value MSE change: own solid, common dashed')
    axes[0,2].set_title('Common bank KL to frozen native mean target')
    axes[0,0].legend(fontsize=8)
    for j,axis in enumerate(axes[0]):
        axis.set_xlabel('Student updates');axis.grid(alpha=.2)
        axis.set_ylabel(('Delta KL','Delta MSE','KL')[j])
        if j<2:axis.axhline(0,color='#bbb',linewidth=.7)
    common_kls=[r['common']['summary']['all']['policy_kl_to_frozen_mean_targets']['native'] for a in ARMS for r in analysis['training'][a]['curve']]
    axes[0,2].set_ylim(min(common_kls)-.005,max(common_kls)+.005)
    for j,mode in enumerate(('greedy','native')):
        axis=axes[1,j]
        for a,arm in enumerate(ARMS):
            x=np.array([0,1])+(a-1)*.23
            rows=[analysis['panels'][f'{arm}_{s}_{mode}'] for s in (100,500)]
            scores=np.array([r['win_rate_score']*100 for r in rows])
            lo=np.array([r['paired_bootstrap']['lower']*100 for r in rows]);hi=np.array([r['paired_bootstrap']['upper']*100 for r in rows])
            axis.errorbar(x,scores,yerr=np.stack((scores-lo,hi-scores)),fmt='o',capsize=4,color=COLORS[arm],label=LABELS[arm])
        axis.set_xticks([0,1],['100 updates','500 updates']);axis.set_ylabel('Score (%)')
        axis.set_title(f'Common {mode} vs frozen16500: DEV50 pairs')
        axis.axhline(50,color='#bbb',linestyle=':');axis.set_ylim(0,100);axis.grid(axis='y',alpha=.2)
    secs=[analysis['collection'][a]['wall_seconds_including_validation'] for a in ARMS]
    axes[1,2].bar([LABELS[a] for a in ARMS],np.array(secs)/60,color=[COLORS[a] for a in ARMS])
    axes[1,2].set_title('200-game collection + full checked replay')
    axes[1,2].set_ylabel('Wall time (minutes)');axes[1,2].tick_params(axis='x',labelsize=8)
    for i,seconds in enumerate(secs):axes[1,2].text(i,seconds/60+.1,f'{seconds/60:.2f}',ha='center',fontsize=9)
    fig.suptitle('Frozen16500 teacher | S1024 | latest-only data | detached value + separate clipping',fontsize=12)
    fig.savefig(ART/'curves.png',dpi=180);fig.savefig(ART/'curves.pdf')
    print('Saved curves.png / curves.pdf')

    confirm=json.loads((ART/'confirmation.json').read_text())
    fig2,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for i,arm in enumerate(('native','cpw')):
        row=confirm['results'][arm];score=row['win_rate_score']*100;ci=row['paired_bootstrap']
        ax[0].errorbar(i,score,yerr=[[score-ci['lower']*100],[ci['upper']*100-score]],fmt='o',color=COLORS[arm],capsize=5)
        ax[0].text(i,score+1,f'{score:.2f}%',ha='center',fontsize=10)
    ax[0].set_xticks([0,1],['Determinized data','CPW data']);ax[0].set_ylim(30,70)
    ax[0].axhline(50,color='#aaa',linestyle=':');ax[0].set_ylabel('Score (%)')
    ax[0].set_title('Independent100 pairs vs frozen16500')
    d=confirm['difference_cpw_minus_native'];score=d['difference']*100
    ax[1].errorbar(score,0,xerr=[[score-d['lower']*100],[d['upper']*100-score]],fmt='o',color=COLORS['cpw'],capsize=5)
    ax[1].axvline(0,color='#aaa',linestyle=':');ax[1].set_xlim(-15,15)
    ax[1].set_yticks([0],['CPW - control']);ax[1].set_xlabel('Paired score difference (percentage points)')
    ax[1].set_title(f"{score:+.2f} pp, 95% CI [{d['lower']*100:+.2f}, {d['upper']*100:+.2f}]")
    for axis in ax:axis.grid(alpha=.2)
    fig2.suptitle('Student100 | common native S1024 | separate clipping / detached value',fontsize=11)
    fig2.savefig(ART/'confirmation.png',dpi=180);fig2.savefig(ART/'confirmation.pdf')
    print('Saved confirmation.png / confirmation.pdf')
