from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from splendor.actions import Pay, ReserveDeck, ReturnTokens
from splendor.compact_trace import decode_request
from splendor.engine import GameEngine
from splendor.model import Tier, TokenVector
from splendor.recommend import LocalAdvisor, decision_identity, describe_action, read_decision
from splendor.replay import to_primitive


def test_jsonl_requires_explicit_position_and_ignores_outcome_fields(tmp_path):
    request = GameEngine.new(7).decision()
    row = {'request':to_primitive(request),'outcome_value':-1,'action':{'irrelevant':True}}
    path = tmp_path/'rows.jsonl'
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='requires'):
        read_decision(path)
    assert read_decision(path,decision_index=0) == request
    with pytest.raises(ValueError,match='outside'):
        read_decision(path,decision_index=1)


def test_greedy_recommends_without_mutating_request_or_game(catalog):
    engine = GameEngine.new(7)
    request = engine.decision()
    before = engine.current_state_digest
    count = len(request.legal_actions)
    class FakeEvaluator:
        def policy_value(self,requests):
            assert requests == [request]
            return [(np.arange(count,dtype=float),.25)]
    advisor = object.__new__(LocalAdvisor)
    advisor.catalog = catalog
    advisor.mode = 'greedy'
    advisor.evaluator = FakeEvaluator()
    advisor.search_policy = None
    advisor.checkpoint = Path('test-only.pt')
    advisor.checkpoint_sha256 = 'test-only'
    result = advisor.recommend(request,top_k=2)
    assert result['recommended_action'] == to_primitive(request.legal_actions[-1])
    assert result['recommended_candidate_index'] == count-1
    assert len(result['candidates']) == 2
    assert result['value_actor_raw'] == .25
    assert result['submitted_action'] is False
    assert engine.current_state_digest == before
    assert result['decision_sha256'] == decision_identity(request)


def test_nonfinite_inference_is_rejected(catalog):
    request = GameEngine.new(7).decision()
    class FakeEvaluator:
        def policy_value(self,requests):
            return [(np.full(len(request.legal_actions),np.nan),0.)]
    advisor = object.__new__(LocalAdvisor)
    advisor.evaluator = FakeEvaluator()
    with pytest.raises(ValueError,match='non-finite'):
        advisor.recommend(request)


def test_action_descriptions_preserve_payment_gold_and_hidden_reservation(catalog):
    assert describe_action(Pay(TokenVector(blue=1,gold=2)),catalog) == '支付蓝1、金2'
    assert describe_action(ReturnTokens(TokenVector(white=2)),catalog) == '归还白2'
    assert describe_action(ReserveDeck(Tier.TWO),catalog) == '从2级牌堆盲保留1张'


def test_advice_identity_changes_when_candidate_order_changes():
    raw = to_primitive(GameEngine.new(7).decision())
    before = decode_request(raw)
    raw['legal_actions'].reverse()
    after = decode_request(raw)
    assert decision_identity(before) != decision_identity(after)
