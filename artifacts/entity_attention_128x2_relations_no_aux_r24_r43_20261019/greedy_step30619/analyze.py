"""Verify the requested 30619-step greedy panel and compare shared R43 openers."""
from pathlib import Path

from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.paired_statistics import (paired_bootstrap_difference, paired_bootstrap_mean,
                                       serialized_pair_scores)

ART = Path(__file__).resolve().parent
ROOT = ART.parents[2]


def main():
    plan = read_json(ART / 'plan.json')
    config = read_json(ART / 'evaluation_config.json')
    result = read_json(ART / 'evaluation.json')
    reference = read_json(ROOT / plan['reference_panel'])
    assert result['config'] == config and result['completed_pairs'] == plan['pairs'] == 1000
    assert result['valid_games'] == 2000
    assert result['checkpoint_identity']['policy_a']['sha256'] == plan['candidate_sha256']
    assert result['checkpoint_identity']['policy_b'] == reference['checkpoint_identity']['policy_b']
    assert result['config']['policy_seed_base'] == reference['config']['policy_seed_base']
    assert result['config']['policy_b'] == reference['config']['policy_b']
    seeds, scores = serialized_pair_scores(result)
    ref_seeds, ref_scores = serialized_pair_scores(reference)
    assert seeds == tuple(config['game_seeds']) == ref_seeds
    interval = paired_bootstrap_mean(scores, resamples=plan['bootstrap_resamples'], seed=plan['bootstrap_seed']).to_dict()
    assert interval == result['paired_bootstrap']
    for pair in result['pairs']:
        assert [game['first_player'] for game in pair['games']] == [0, 1]
        for game in pair['games']:
            for side in ('policy_a', 'policy_b'):
                stats = game[f'{side}_stats']
                assert not stats.get('simulations', 0) and not stats.get('determinizations', 0)
    difference = paired_bootstrap_difference(seeds, scores, ref_seeds, ref_scores,
        resamples=plan['bootstrap_resamples'], seed=plan['difference_bootstrap_seed']).to_dict()
    summary = {'status': 'complete', 'candidate_step': 30619, 'reference_step': 37423,
               'pairs': 1000, 'score': result['win_rate_score'], 'ci95': interval,
               'reference_score': reference['win_rate_score'], 'step30619_minus_step37423': difference,
               'candidate_sha256': plan['candidate_sha256'], 'evaluation_sha256': sha256(ART / 'evaluation.json'),
               'reference_sha256': sha256(ROOT / plan['reference_panel']),
               'scope': plan['scope'], **{key: result[key] for key in ('a_wins', 'draws', 'a_losses',
                   'exceptions', 'illegal_actions', 'engineering_truncations', 'invariant_failures', 'policy_cycle_draws')}}
    write_json(ART / 'summary.json', summary)
    lines = ['# 新128×2：30619步对R43的1000pair greedy', '',
             f"完成1000pair／2000局；{result['a_wins']}胜、{result['draws']}和、{result['a_losses']}负。",
             f"得分 **{100*summary['score']:.3f}%**，95%整pair区间 **[{100*interval['lower']:.3f}%, {100*interval['upper']:.3f}%]**。", '',
             '| 节点 | 对R43得分 |', '| --- | ---: |',
             f"| 30619 | {100*summary['score']:.3f}% |",
             f"| 37423 | {100*summary['reference_score']:.3f}% |", '',
             f"30619减37423的共同对手配对得分差：**{100*difference['estimate']:+.3f}pp**，95%区间 "
             f"**[{100*difference['lower']:+.3f}, {100*difference['upper']:+.3f}]pp**。", '',
             '差值为相同开局、相同R43对手下的配对比较，不是两个候选直接互打。单训练seed、已查看DEV开局的事后节点补测。', '',
             f"异常／非法动作／工程截断／不变量失败／循环裁和：{result['exceptions']}/{result['illegal_actions']}/"
             f"{result['engineering_truncations']}/{result['invariant_failures']}/{result['policy_cycle_draws']}。", '']
    (ART / 'results.md').write_text('\n'.join(lines))
    print({'score': summary['score'], 'ci95': interval, 'paired_difference': difference}, flush=True)


if __name__ == '__main__':
    main()
