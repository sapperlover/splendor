"""Reproduce captured live roots without opening a browser or submitting actions."""
import json
from pathlib import Path

from splendor.bga_live import ReservationHistory, request_from_snapshot
from splendor.recommend import decision_identity
from splendor.replay import to_primitive
from splendor.search_policy import determinize_request

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def main():
    results=[]
    for step in (2,4,6,8):
        case=HERE/'snapshots'/f'step{step}'
        snapshot=json.loads((case/'snapshot.json').read_text())
        history=ReservationHistory(**json.loads((case/'history.json').read_text()))
        saved_request=json.loads((case/'request.json').read_text())
        result=json.loads((case/'recommendation.json').read_text())
        request,catalog,mapping=request_from_snapshot(snapshot,history)
        assert to_primitive(request)==saved_request['request']
        assert mapping==saved_request['bga_mapping']
        assert decision_identity(request)==result['decision_sha256']
        assert result['recommended_action'] in [to_primitive(a) for a in request.legal_actions]
        assert result['submitted_action'] is False
        for seed in (0,123):
            assert determinize_request(request,seed=seed,catalog=catalog,engine_mode='fast').decision()==request
        results.append({'step':step,'candidates':len(request.legal_actions),'decision_sha256':result['decision_sha256'],'recommendation':result['description']})
    print(json.dumps({'captured_roots_verified':results,'submitted_action':False},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
