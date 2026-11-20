"""Probability/backup and private-information boundaries for sampled chance PUCT."""
from collections import Counter
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from splendor.actions import ReserveDeck, ReserveMarket, BuyReserved
from splendor.actor_critic import ActorCritic
from splendor.benchmark_search import fixed_requests
from splendor.catalog import load_default_catalog
from splendor.engine import GameEngine
from splendor.evaluation import SerializablePolicySpec
from splendor.mcts_policy import MCTSConfig
from splendor.model import Tier, DecisionPhase
from splendor.native_policy import NativeChancePUCTSearchPolicy, NativeBatchRunner, ChanceWideningConfig
from splendor.search_inference import FrozenSearchInference
from splendor.search_policy import _request_public_word
from splendor._state import PendingTurn

native = pytest.importorskip("splendor._native_search")


def conserve(snapshot, catalog):
    cards = [x for tier in snapshot["decks"] for x in tier]
    cards += [x for tier in snapshot["market"] for x in tier]
    for player in snapshot["players"]:
        cards += player["purchased"] + [r[0] for r in player["reserved"]]
    assert sorted(cards) == list(range(90))
    tokens = np.asarray(snapshot["supply"]) + sum(np.asarray(p["tokens"]) for p in snapshot["players"])
    assert tokens.tolist() == [4, 4, 4, 4, 4, 5]
    for tier, deck in enumerate(snapshot["decks"]):
        assert all(int(catalog.cards[i].tier) - 1 == tier for i in deck)


def test_random_refill_samples_uniformly_and_backs_up_average_not_best_outcome():
    catalog = load_default_catalog();nc = native.Catalog(catalog)
    engine = GameEngine.new(42, first_player=0, mode="checked")
    request = engine.decision()
    initial = native.NativeGame(nc, engine._debug_state()).snapshot()
    index = next(i for i, a in enumerate(request.legal_actions) if isinstance(a, ReserveMarket)
                 and catalog.card(a.card_id).tier == Tier.ONE)
    config = MCTSConfig(simulations=1024, determinizations=1, seed=55, max_simulation_transitions=1)
    job = native.SearchJob(nc, request, config, _request_public_word(request), 99, False, 0.0, True)
    logits = [-1000.0] * len(request.legal_actions);logits[index] = 1000.0
    draws = Counter();rewards = []
    while not job.done():
        for item in job.pending():
            if item["world"] == -1:
                job.submit(-1, logits, 0.0);job.initialize_worlds(request);continue
            assert item["kind"] == "value_only"
            snapshot = job.debug_leaf_state(0);conserve(snapshot, catalog)
            new = set(snapshot["market"][0]) - set(initial["market"][0])
            assert len(new) == 1
            card = new.pop();draws[card] += 1
            reward = 0.25 if card % 2 == 0 else 0.75;rewards.append(reward)
            # Leaf actor is the opponent: the root's positive payoff is a negative critic value.
            assert snapshot["actor"] == 1
            job.submit(0, (), -reward)
    pool = set(initial["decks"][0]);assert set(draws) == pool
    expectation = 1024 / len(pool)
    assert max(abs(count - expectation) for count in draws.values()) < 6 * np.sqrt(expectation)
    result = job.result();assert result["root_visit_counts"][index] == 1024
    diagnostic = job.chance_diagnostics()
    assert diagnostic["root_hidden_samples"] == diagnostic["chance_draws"] == 1024
    assert sorted(diagnostic["root_outcome_visits"][index]) == sorted(draws.values())
    assert diagnostic["root_value_sums"][index] == sum(rewards)
    assert 0.25 < diagnostic["root_value_sums"][index] / 1024 < 0.75


def test_information_keys_hide_opponent_cards_and_future_order_but_keep_own_private_cards():
    catalog = load_default_catalog();nc = native.Catalog(catalog)
    engine = GameEngine.new(42, first_player=0, mode="checked")
    engine.apply(ReserveDeck(Tier.ONE));state = engine._debug_state()
    old = state.players[0].reserved_cards[0].card_id;replacement = state.decks.tier_one[-1]
    reserved = replace(state.players[0].reserved_cards[0], card_id=replacement)
    player = replace(state.players[0], reserved_cards=(reserved,))
    other = replace(state, players=(player, state.players[1]),
                    decks=replace(state.decks, tier_one=(*state.decks.tier_one[:-1], old)))
    assert native.information_key(nc, state, 1) == native.information_key(nc, other, 1)
    assert native.information_key(nc, state, 0) != native.information_key(nc, other, 0)
    reordered = replace(state, decks=replace(state.decks, tier_one=tuple(reversed(state.decks.tier_one))))
    for observer in (0, 1):
        assert native.information_key(nc, state, observer) == native.information_key(nc, reordered, observer)
    # A private reserved purchase is not exposed to the other player before it becomes public.
    first_pending = replace(state, current_player=0, phase=DecisionPhase.PAYMENT, pending=PendingTurn(BuyReserved(old)))
    other_pending = replace(other, current_player=0, phase=DecisionPhase.PAYMENT, pending=PendingTurn(BuyReserved(replacement)))
    assert native.information_key(nc, first_pending, 1) == native.information_key(nc, other_pending, 1)
    assert native.information_key(nc, first_pending, 0) != native.information_key(nc, other_pending, 0)


def test_existing_blind_cards_are_resampled_each_simulation_without_duplicate_cards():
    catalog = load_default_catalog();nc = native.Catalog(catalog)
    engine = GameEngine.new(42, first_player=0, mode="checked");engine.apply(ReserveDeck(Tier.ONE))
    request = engine.decision()
    config = MCTSConfig(simulations=32, determinizations=1, seed=17, max_simulation_transitions=1)
    job = native.SearchJob(nc, request, config, _request_public_word(request), 18, False, 0.0, True)
    seen = set()
    while not job.done():
        for item in job.pending():
            if item["world"] == -1:
                logits = [-1000.0] * len(request.legal_actions);logits[0] = 1000.0
                job.submit(-1, logits, 0.0);job.initialize_worlds(request)
            else:
                snapshot = job.debug_leaf_state(0);conserve(snapshot, catalog)
                seen.add(snapshot["players"][1]["reserved"][0][0])
                job.submit(0, (), 0.0)
    assert len(seen) > 8 and job.chance_diagnostics()["root_hidden_samples"] == 32


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("widening", [None, ChanceWideningConfig()])
def test_batched_chance_search_equals_per_leaf_path(position, widening):
    torch.set_num_threads(1)
    model = ActorCritic().eval()
    for param in model.parameters():param.data.zero_()
    request = fixed_requests()[position][1]
    config = MCTSConfig(simulations=33, determinizations=1, seed=91)
    evaluator = FrozenSearchInference(model)
    policy = NativeChancePUCTSearchPolicy(model, config=config, evaluator=evaluator, chance_widening=widening)
    expected = policy.search(request, add_root_noise=True, sampling_seed=92)
    job = policy.start_job(request, add_root_noise=True, sampling_seed=92)
    runner = NativeBatchRunner(evaluator, policy.catalog, capacity=4, buckets=(16, 32, 48, 64))
    while not job.done:assert runner.run([job]) > 0
    assert asdict(job.result) == asdict(expected)
    assert job.core.chance_diagnostics()["root_hidden_samples"] == 33


def test_chance_backend_configuration_roundtrips_and_rejects_independent_world_budget():
    spec = SerializablePolicySpec.mcts_checkpoint("chance", "test.pt", search_backend="native_chance",
        search_config={"simulations": 8, "determinizations": 1})
    assert SerializablePolicySpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(ValueError, match="determinizations=1"):
        replace(spec, search_config={"simulations": 8, "determinizations": 2})
    with pytest.raises(ValueError, match="determinizations=1"):
        NativeChancePUCTSearchPolicy(ActorCritic(), config=MCTSConfig(simulations=8, determinizations=2))


@pytest.mark.parametrize('batch_size', [1, 8, 32])
def test_single_tree_batch_distinct_hidden_samples_and_exact_backup(batch_size):
    catalog=load_default_catalog();nc=native.Catalog(catalog)
    engine=GameEngine.new(42,first_player=0,mode='checked')
    engine.apply(ReserveDeck(Tier.ONE));request=engine.decision()
    config=MCTSConfig(simulations=67,determinizations=1,seed=17,max_simulation_transitions=1)
    job=native.SearchJob(nc,request,config,_request_public_word(request),18,False,0.,True,
                         leaf_batch_size=batch_size)
    seen=set();ids=set();root_actor=0  # Native requests are normalized to the actor's seat.
    while not job.done():
        pending=job.pending()
        assert pending and len(pending)<=batch_size
        for item in pending:
            if item['world']==-1:
                job.submit(-1,[0.]*len(request.legal_actions),0.);job.initialize_worlds(request)
                continue
            simulation=item['simulation'];assert simulation not in ids;ids.add(simulation)
            state=job.debug_leaf_state(0,simulation);conserve(state,catalog)
            seen.add(state['players'][1]['reserved'][0][0])
            value=(simulation%7-3)/4
            # A one-transition horizon has exactly one root edge in its path.
            before=list(job.chance_diagnostics()['root_value_sums'])
            job.submit(0,(),value,simulation)
            after=job.chance_diagnostics()['root_value_sums']
            assert sum(after)-sum(before)==pytest.approx(value if state['actor']==root_actor else -value)
    result=job.result();diag=job.chance_diagnostics()
    assert ids==set(range(67)) and len(seen)>8
    assert sum(result['root_visit_counts'])==diag['root_hidden_samples']==67
    assert diag['virtual_visits']==diag['pending_simulations']==0
    assert diag['peak_pending_simulations']==min(batch_size,67)


@pytest.mark.parametrize('position', [0,1,2])
def test_single_tree_multileaf_batch_transport_and_collision_accounting(position):
    from splendor.search_inference import FixedValueSearchInference
    request=fixed_requests()[position][1]
    evaluator=FixedValueSearchInference()
    policy=NativeChancePUCTSearchPolicy(None,evaluator=evaluator,
        config=MCTSConfig(simulations=257,determinizations=1,seed=91),leaf_batch_size=16)
    job=policy.start_job(request)
    runner=NativeBatchRunner(evaluator,policy.catalog,capacity=7,buckets=(16,32,48,64))
    while not job.done:assert runner.run([job])>0
    diag=job.core.chance_diagnostics()
    assert job.result.simulations==sum(job.result.root_visit_counts)==257
    assert diag['root_hidden_samples']==257
    assert diag['pending_simulations']==diag['virtual_visits']==0
    assert diag['leaf_collisions']>=0
    assert evaluator.stats.positions+diag['leaf_collisions']<=258
    assert any(size>1 for size in evaluator.stats.batch_sizes)


def test_chance_leaf_batch_validation():
    with pytest.raises(ValueError,match='positive integer'):
        NativeChancePUCTSearchPolicy(ActorCritic(),leaf_batch_size=0)
    catalog=load_default_catalog();request=fixed_requests()[0][1]
    with pytest.raises(ValueError,match='batching requires'):
        native.SearchJob(native.Catalog(catalog),request,MCTSConfig(simulations=8,determinizations=1),
                         123,0,False,0.,leaf_batch_size=8)


def test_colliding_leaves_share_one_forward_but_backup_each_sample_once():
    from splendor.actions import TakeDistinct
    catalog=load_default_catalog();request=GameEngine.new(42,first_player=0).decision()
    index=next(i for i,a in enumerate(request.legal_actions) if isinstance(a,TakeDistinct))
    job=native.SearchJob(native.Catalog(catalog),request,
        MCTSConfig(simulations=17,determinizations=1,seed=91),123,0,False,0.,True,leaf_batch_size=8)
    logits=[-1000.]*len(request.legal_actions);logits[index]=1000.
    job.submit(-1,logits,0.);job.initialize_worlds(request)
    pending=job.pending();assert len(pending)==1
    assert job.chance_diagnostics()['leaf_collisions']==7
    row=pending[0];assert row['simulation']==0
    assert job.debug_leaf_state(0,0)['actor']==1
    job.submit(0,[0.]*len(row['encoded']['candidate_features']),-.5,0)
    diag=job.chance_diagnostics()
    assert job.stats()['simulations']==8
    assert diag['root_value_sums'][index]==4.
    while not job.done():
        for row in job.pending():
            job.submit(0,() if row['kind']=='value_only' else [0.]*len(row['encoded']['candidate_features']),
                       0.,row['simulation'])
    assert sum(job.result()['root_visit_counts'])==17
    assert job.chance_diagnostics()['virtual_visits']==0


def test_batched_chance_repeated_seed_is_deterministic():
    from splendor.search_inference import FixedValueSearchInference
    evaluator=FixedValueSearchInference()
    policy=NativeChancePUCTSearchPolicy(None,evaluator=evaluator,
        config=MCTSConfig(simulations=193,determinizations=1,seed=91),leaf_batch_size=16)
    request=fixed_requests()[1][1]
    assert asdict(policy.search(request))==asdict(policy.search(request))


def run_one_draw(simulations, *, blind_opponent=False, alpha=0.5):
    catalog = load_default_catalog();nc = native.Catalog(catalog)
    engine = GameEngine.new(42, first_player=0, mode="checked")
    if blind_opponent:engine.apply(ReserveDeck(Tier.ONE))
    request = engine.decision()
    initial = native.NativeGame(nc, engine._debug_state()).snapshot()
    index = next(i for i, a in enumerate(request.legal_actions) if isinstance(a, ReserveMarket)
                 and catalog.card(a.card_id).tier == Tier.ONE)
    config = MCTSConfig(simulations=simulations, determinizations=1, seed=55, max_simulation_transitions=1)
    job = native.SearchJob(nc, request, config, _request_public_word(request), 99, False, 0.0, True, 1.0, alpha)
    logits = [-1000.0] * len(request.legal_actions);logits[index] = 1000.0
    draws = Counter();rewards=[];hidden=set()
    while not job.done():
        for item in job.pending():
            if item["world"] == -1:
                job.submit(-1, logits, 0.0);job.initialize_worlds(request);continue
            snapshot=job.debug_leaf_state(0);conserve(snapshot,catalog)
            added=set(snapshot["market"][0])-set(initial["market"][0]);assert len(added)==1
            card=added.pop();draws[card]+=1
            reward=0.25 if card%2==0 else 0.75;rewards.append(reward)
            if blind_opponent:
                reserved=snapshot["players"][1]["reserved"][0][0]
                assert reserved!=card;hidden.add(reserved)
            job.submit(0,(),-reward)
    diagnostic=job.chance_diagnostics()
    assert diagnostic["chance_pw_fresh_samples"]+diagnostic["chance_pw_reuses"]==simulations
    assert diagnostic["root_value_sums"][index]==sum(rewards)
    assert job.result()["root_visit_counts"][index]==simulations
    return diagnostic,index,draws,hidden


def test_chance_widening_support_grows_and_reuses_equal_probability_outcomes():
    small,index,_,_=run_one_draw(16)
    large,_,draws,_=run_one_draw(1024)
    first=small["root_chance_pools"][index][0];last=large["root_chance_pools"][index][0]
    assert len(small["root_chance_pools"][index])==len(large["root_chance_pools"][index])==1
    assert 1<len(first["outcomes"])<=4
    assert len(first["outcomes"])<len(last["outcomes"])<=32
    assert set(first["outcomes"]).issubset(last["outcomes"])
    assert set(draws)==set(last["outcomes"])
    assert large["chance_pw_reuses"]>large["chance_pw_fresh_samples"]
    assert large["root_hidden_samples"]==1024
    # Retain all physical outcomes at k=1 with a near-one exponent, then verify
    # no value-dependent preference in repeated draws from the admitted support.
    complete,_,counts,_=run_one_draw(4096,alpha=0.99)
    pool=complete["root_chance_pools"][index][0]
    assert len(pool["outcomes"])==pool["pool_size"]==36
    expected=4096/36
    assert max(abs(n-expected) for n in counts.values())<6*np.sqrt(expected)


def test_chance_widening_conditions_on_remaining_pool_and_keeps_hidden_resampling():
    diagnostic,index,_,hidden=run_one_draw(256,blind_opponent=True)
    assert len(hidden)>8 and diagnostic["root_hidden_samples"]==256
    pools=diagnostic["root_chance_pools"][index]
    assert len(pools)==len(hidden) and sum(p["visits"] for p in pools)==256
    for pool in pools:
        assert len(pool["outcomes"])<=min(pool["pool_size"],int(np.ceil(np.sqrt(pool["visits"]))))


def test_chance_widening_configuration_roundtrips_and_rejects_wrong_backend():
    spec=SerializablePolicySpec.mcts_checkpoint("chance", "test.pt", search_backend="native_chance",
        search_config={"simulations":8,"determinizations":1},chance_widening={"k":1.0,"alpha":0.5})
    assert SerializablePolicySpec.from_dict(spec.to_dict())==spec
    assert "chance_widening" not in replace(spec,chance_widening=None).to_dict()
    with pytest.raises(ValueError,match="native_chance"):replace(spec,search_backend="native")
    with pytest.raises(ValueError,match="native_chance"):
        SerializablePolicySpec.from_dict(dict(spec.to_dict(),kind="greedy_checkpoint",search_backend="python"))
    for kwargs in ({"k":0},{"k":float("nan")},{"alpha":0},{"alpha":1},{"alpha":True}):
        with pytest.raises(ValueError):ChanceWideningConfig(**kwargs)
