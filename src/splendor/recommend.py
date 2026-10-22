"""Read a player-safe decision and recommend actions without applying them."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .actions import (BuyMarket, BuyReserved, ClaimNoble, Pay, ReserveDeck,
                      ReserveMarket, ReturnTokens, TakeDistinct, TakePair)
from .catalog import Catalog, load_default_catalog
from .compact_trace import decode_request
from .contracts import DecisionRequest
from .mcts_policy import MCTSConfig
from .opponent_pool import load_frozen_actor_critic_checkpoint
from .replay import to_primitive
from .search_inference import FrozenSearchInference, InferenceConfig

COLOR_NAMES = {'white':'白', 'blue':'蓝', 'green':'绿', 'red':'红', 'black':'黑', 'gold':'金'}
DEFAULT_SIMULATIONS = 16384


def decision_identity(request: DecisionRequest) -> str:
    """Bind a recommendation to the exact observed state and candidate order."""
    raw = json.dumps(to_primitive(request), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode()).hexdigest()


def describe_action(action, catalog: Catalog) -> str:
    if isinstance(action, TakeDistinct):
        return '拿取' + '、'.join(COLOR_NAMES[c.value] for c in action.colors) + '各1枚'
    if isinstance(action, TakePair):
        return f'拿取{COLOR_NAMES[action.color.value]}色筹码2枚'
    if isinstance(action, ReserveDeck):
        return f'从{int(action.tier)}级牌堆盲保留1张'
    if isinstance(action, (BuyMarket, BuyReserved, ReserveMarket)):
        card = catalog.card(action.card_id)
        verb = {BuyMarket:'购买市场牌', BuyReserved:'购买保留牌', ReserveMarket:'保留市场牌'}[type(action)]
        cost = '、'.join(f'{COLOR_NAMES[c]}{n}' for c,n in zip(COLOR_NAMES, card.cost) if n)
        return f'{verb} {card.card_id}（{int(card.tier)}级，{COLOR_NAMES[card.bonus.value]}加成，{card.points}分，原价{cost}）'
    if isinstance(action, (Pay, ReturnTokens)):
        tokens = '、'.join(f'{COLOR_NAMES[c]}{n}' for c,n in zip(COLOR_NAMES, action.tokens) if n) or '0枚'
        return ('支付' if isinstance(action, Pay) else '归还') + tokens
    if isinstance(action, ClaimNoble):
        noble = catalog.noble(action.noble_id)
        need = '、'.join(f'{COLOR_NAMES[c]}{n}' for c,n in zip(COLOR_NAMES,noble.requirement) if n)
        return f'领取贵族 {noble.noble_id}（{need}，{noble.points}分）'
    raise TypeError(f'Unsupported action: {action!r}')


class LocalAdvisor:
    """Model inference/search only; no engine application or browser action."""

    def __init__(self, checkpoint: Path, *, device: str = 'cpu', mode: str = 'greedy',
                 simulations: int = DEFAULT_SIMULATIONS, determinizations: int = 1, seed: int = 0,
                 catalog: Catalog | None = None):
        if mode not in ('greedy', 'mcts'):
            raise ValueError('mode must be greedy or mcts')
        self.checkpoint = checkpoint.resolve(strict=True)
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.catalog = catalog or load_default_catalog()
        self.mode = mode
        self.model = load_frozen_actor_critic_checkpoint(self.checkpoint, device=device)
        self.evaluator = FrozenSearchInference(self.model, device=device,
                                              config=InferenceConfig(mode='eager',actor_projection=False))
        self.search_policy = None
        if mode == 'mcts':
            from .native_policy import NativePUCTSearchPolicy
            config = MCTSConfig(simulations=simulations, determinizations=determinizations,
                                seed=seed, root_noise_fraction=0., forced_playouts=0.,
                                policy_target_pruning=False)
            self.search_policy = NativePUCTSearchPolicy(self.model, evaluator=self.evaluator,
                                                        catalog=self.catalog, config=config, device=device)

    def recommend(self, request: DecisionRequest, *, top_k: int = 3) -> dict:
        if type(top_k) is not int or top_k < 1:
            raise ValueError('top_k must be positive')
        started = time.perf_counter()
        identity = decision_identity(request)
        logits, value = self.evaluator.policy_value([request])[0]
        logits = np.asarray(logits, dtype=np.float64)
        if logits.shape != (len(request.legal_actions),) or not np.isfinite(logits).all() or not math.isfinite(value):
            raise ValueError('non-finite or incorrectly sized model output')
        probabilities = np.exp(logits-logits.max())
        probabilities /= probabilities.sum()
        result = None if self.search_policy is None else self.search_policy.search(
            request, add_root_noise=False, temperature=0.)
        if result is None:
            ranking = sorted(range(len(logits)), key=lambda i:(-probabilities[i],i))
            chosen = ranking[0]
        else:
            chosen = result.chosen_candidate_index
            ranking = sorted(range(len(logits)), key=lambda i:(-result.root_visit_counts[i],-result.root_network_prior[i],i))
            assert ranking[0] == chosen
        candidates = []
        for i in ranking[:top_k]:
            row = {'candidate_index':i,'action':to_primitive(request.legal_actions[i]),
                   'description':describe_action(request.legal_actions[i],self.catalog),
                   'policy_probability':float(probabilities[i])}
            if result is not None:
                row['visits'] = result.root_visit_counts[i]
                row['visit_share'] = (row['visits']/result.simulations) if result.simulations else None
                sums = result.per_world_root_value_sums
                row['search_q_actor'] = (sum(world[i] for world in sums)/row['visits']) if sums and row['visits'] else None
            candidates.append(row)
        assert identity == decision_identity(request)
        return {'schema':'splendor-read-only-recommendation-v1','decision_sha256':identity,
                'phase':request.observation.phase.value,'mode':self.mode,
                'checkpoint':str(self.checkpoint),'checkpoint_sha256':self.checkpoint_sha256,
                'catalog_fingerprint':self.catalog.data_fingerprint,
                'recommended_candidate_index':chosen,'recommended_action':to_primitive(request.legal_actions[chosen]),
                'description':describe_action(request.legal_actions[chosen],self.catalog),
                'value_actor_raw':float(value),'candidate_count':len(request.legal_actions),
                'candidates':candidates,'simulations':0 if result is None else result.simulations,
                'elapsed_seconds':time.perf_counter()-started,'submitted_action':False}


def read_decision(path: Path, *, decision_index: int | None = None) -> DecisionRequest:
    """Read native DTO JSON, an archived row, or one explicit JSONL row."""
    text = path.read_text(encoding='utf-8')
    if path.suffix == '.jsonl':
        if decision_index is None or decision_index < 0:
            raise ValueError('JSONL input requires --decision-index >= 0')
        rows = text.splitlines()
        if decision_index >= len(rows):
            raise ValueError('decision index is outside input file')
        value = json.loads(rows[decision_index])
    else:
        if decision_index is not None:
            raise ValueError('--decision-index is only valid with JSONL input')
        value = json.loads(text)
    return decode_request(value['request'] if 'request' in value else value)


def main(argv=None):
    parser = argparse.ArgumentParser(description='读取局面并推荐动作；不会执行对局动作。')
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--decision-index',type=int)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--mode',choices=('greedy','mcts'),default='greedy')
    parser.add_argument('--simulations',type=int,default=DEFAULT_SIMULATIONS)
    parser.add_argument('--determinizations',type=int,default=1)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--top-k',type=int,default=3)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    request = read_decision(args.request,decision_index=args.decision_index)
    advisor = LocalAdvisor(args.checkpoint,device=args.device,mode=args.mode,
                           simulations=args.simulations,determinizations=args.determinizations,seed=args.seed)
    result = advisor.recommend(request,top_k=args.top_k)
    raw = json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(raw,encoding='utf-8')
    print(raw,end='')


if __name__ == '__main__':
    main()
