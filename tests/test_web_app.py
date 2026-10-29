from __future__ import annotations

from pathlib import Path

import pytest
import torch

from splendor.actor_critic import ActorCritic
import splendor.web_app as web_app
from splendor.contracts import PublicReservedCard
from splendor.actions import ReserveDeck
from splendor.model import ReservationSource, Tier
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativePUCTSearchPolicy
from splendor.heuristic_policy import ScorePolicy
from splendor.search_policy import PolicyValueSearchPolicy, SearchConfig
from splendor.web_app import (
    WebGameService,
    _action_history_text,
    _opponent_reservations,
)


def _service(tmp_path: Path) -> WebGameService:
    return WebGameService(
        failure_dir=tmp_path / "failures",
        checkpoint_dir=tmp_path / "play_checkpoints",
        replay_dir=tmp_path / "replays",
    )


def test_new_web_game_exposes_observation_and_legal_candidates_only(
    tmp_path: Path,
) -> None:
    payload = _service(tmp_path).create_game(
        opponent="score", human_first=True, seed=17
    )

    assert payload["phase"] == "main_action"
    assert payload["players"]["self"]["isFirst"] is True
    assert payload["opponent"] == "score"
    assert payload["opponentLabel"] == "ScorePolicy"
    assert len(payload["market"]["1"]) == 4
    assert len(payload["market"]["2"]) == 4
    assert len(payload["market"]["3"]) == 4
    assert payload["legalActions"]
    assert {"index", "kind", "label"} <= payload["legalActions"][0].keys()
    assert "state" not in payload
    assert "rng" not in payload
    assert "replay" not in payload
    assert "decks" not in payload


def test_playable_checkpoint_directory_is_discovered_without_path_exposure(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    checkpoint = service.checkpoint_dir / "step_000000123.pt"
    checkpoint.write_bytes(b"listed but loaded only when selected")

    opponents = service.available_opponents()

    assert [opponent["id"] for opponent in opponents] == [
        "score",
        "development",
        "checkpoint:step_000000123.pt",
    ]
    assert all("/" not in opponent["id"] for opponent in opponents)


@pytest.mark.parametrize("opponent", ["score", "development"])
def test_human_second_advances_one_fixed_policy_turn_before_response(
    tmp_path: Path, opponent: str
) -> None:
    payload = _service(tmp_path).create_game(
        opponent=opponent, human_first=False, seed=23
    )

    assert payload["result"] is None
    assert payload["players"]["self"]["isFirst"] is False
    assert payload["players"]["opponent"]["completedTurns"] == 1
    assert payload["legalActions"]
    assert payload["revision"] >= 1
    assert payload["history"]
    assert payload["history"][0]["actor"] == "opponent"
    assert payload["history"][0]["summary"]


def test_web_action_uses_current_legal_candidate_and_rejects_stale_or_negative_index(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    initial = service.create_game(opponent="development", human_first=True, seed=5)
    updated = service.apply_action(
        initial["gameId"], revision=initial["revision"], action_index=0
    )

    assert updated["revision"] > initial["revision"]
    assert updated["legalActions"] or updated["result"] is not None
    assert {entry["actor"] for entry in updated["history"]} == {
        "self",
        "opponent",
    }
    with pytest.raises(ValueError, match="stale game revision"):
        service.apply_action(
            initial["gameId"], revision=initial["revision"], action_index=0
        )
    with pytest.raises(ValueError, match="actionIndex"):
        service.apply_action(
            initial["gameId"], revision=updated["revision"], action_index=-1
        )


def test_opponent_blind_reservation_serializes_only_tier_and_back() -> None:
    payload = _opponent_reservations(
        (
            PublicReservedCard(
                tier=Tier.THREE,
                source=ReservationSource.DECK,
                card=None,
            ),
        )
    )

    assert payload == [
        {"source": "deck", "hidden": True, "tier": 3, "card": None}
    ]
    assert "card_id" not in repr(payload)


def test_blind_reservation_history_never_contains_drawn_card_identity(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)

    text = _action_history_text(ReserveDeck(Tier.THREE), service.catalog)

    assert text == "从 3 级牌堆盲保留"
    assert not any(card.card_id in text for card in service.catalog.cards)


@pytest.mark.parametrize(
    ("opponent", "human_first", "seed", "message"),
    [
        ("random", True, 1, "opponent"),
        ("score", "yes", 1, "humanFirst"),
        ("score", True, 1.5, "seed"),
    ],
)
def test_web_setup_rejects_unsupported_settings(
    tmp_path: Path,
    opponent: object,
    human_first: object,
    seed: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _service(tmp_path).create_game(
            opponent=opponent,  # type: ignore[arg-type]
            human_first=human_first,  # type: ignore[arg-type]
            seed=seed,  # type: ignore[arg-type]
        )


def test_configured_search_checkpoint_is_listed_with_search_settings(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "local_005000000.pt"
    checkpoint.write_bytes(b"loaded only when selected")
    config = SearchConfig(
        top_k=8,
        depth=6,
        determinizations=1,
        seed=123,
        leaf_value_clip=1.0,
        fold_forced_phases=True,
    )
    service = WebGameService(
        failure_dir=tmp_path / "failures",
        checkpoint_dir=tmp_path / "play_checkpoints",
        search_checkpoint=checkpoint,
        search_config=config,
    )

    search_opponent = service.available_opponents()[-1]

    assert search_opponent == {
        "id": "search",
        "label": (
            "PPO 搜索 · local_005000000 · D6 top8 · "
            "1 determinization · 折叠强制阶段"
        ),
        "kind": "search_checkpoint",
    }


def test_web_game_uses_configured_search_policy_for_opponent_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "search_model.pt"
    checkpoint.write_bytes(b"test checkpoint")
    config = SearchConfig(top_k=8, depth=6, seed=42, fold_forced_phases=True)
    loaded_models: list[tuple[Path, str]] = []
    created_policies: list[tuple[object, SearchConfig, str, object]] = []
    model = object()

    def load_model(path: Path, *, device: str) -> object:
        loaded_models.append((path, device))
        return model

    class FakeSearchPolicy:
        def __init__(
            self,
            policy_model: object,
            *,
            config: SearchConfig,
            device: str,
            catalog: object,
        ) -> None:
            created_policies.append((policy_model, config, device, catalog))
            self.evaluator = self

        def value_only(self, requests):
            return (0.0,) * len(requests)

        def choose(self, request: object) -> object:
            return request.legal_actions[0]  # type: ignore[attr-defined]

    monkeypatch.setattr(web_app, "load_frozen_actor_critic_checkpoint", load_model)
    monkeypatch.setattr(web_app, "PolicyValueSearchPolicy", FakeSearchPolicy)
    service = WebGameService(
        failure_dir=tmp_path / "failures",
        checkpoint_dir=tmp_path / "play_checkpoints",
        replay_dir=tmp_path / "replays",
        search_checkpoint=checkpoint,
        search_config=config,
    )

    payload = service.create_game(opponent="search", human_first=False, seed=37)

    assert payload["opponent"] == "search"
    assert payload["opponentLabel"] == "PPO 搜索 · search_model · D6 top8"
    assert payload["players"]["opponent"]["completedTurns"] == 1
    assert payload["legalActions"]
    assert loaded_models == [(checkpoint.resolve(), "cpu")]
    assert created_policies == [(model, config, "cpu", service.catalog)]


def test_search_configuration_requires_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires search_checkpoint"):
        WebGameService(
            failure_dir=tmp_path / "failures",
            checkpoint_dir=tmp_path / "play_checkpoints",
            search_config=SearchConfig(depth=6, top_k=8),
        )


def test_web_cli_builds_requested_search_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "search_model.pt"
    checkpoint.write_bytes(b"test checkpoint")
    server_args: list[tuple[tuple[str, int], WebGameService]] = []

    class FakeServer:
        def __init__(self, address: tuple[str, int], service: WebGameService) -> None:
            server_args.append((address, service))

        def serve_forever(self) -> None:
            return None

        def server_close(self) -> None:
            return None

    monkeypatch.setattr(web_app, "WebGameServer", FakeServer)

    result = web_app.main(
        [
            "--port", "8766",
            "--search-checkpoint", str(checkpoint),
            "--search-depth", "6",
            "--search-top-k", "8",
            "--search-determinizations", "1",
            "--search-seed", "9200000",
            "--search-leaf-value-clip", "1.0",
            "--search-fold-forced-phases",
        ]
    )

    assert result == 0
    assert server_args[0][0] == ("127.0.0.1", 8766)
    service = server_args[0][1]
    assert service.search_checkpoint == checkpoint.resolve()
    assert service.search_config == SearchConfig(
        depth=6,
        top_k=8,
        determinizations=1,
        seed=9200000,
        leaf_value_clip=1.0,
        fold_forced_phases=True,
    )



def test_web_game_runs_real_search_policy_for_opponent_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "search_model.pt"
    checkpoint.write_bytes(b"test checkpoint")
    model = ActorCritic().eval()
    config = SearchConfig(top_k=1, depth=1, seed=5)
    loaded: list[tuple[Path, str]] = []

    def load_model(path: Path, *, device: str) -> ActorCritic:
        loaded.append((path, device))
        return model

    monkeypatch.setattr(web_app, "load_frozen_actor_critic_checkpoint", load_model)
    service = WebGameService(
        failure_dir=tmp_path / "failures",
        checkpoint_dir=tmp_path / "play_checkpoints",
        replay_dir=tmp_path / "replays",
        search_checkpoint=checkpoint,
        search_config=config,
    )

    payload = service.create_game(opponent="search", human_first=False, seed=43)

    game = service._games[payload["gameId"]]
    assert isinstance(game.opponent, PolicyValueSearchPolicy)
    assert game.opponent.config == config
    assert loaded == [(checkpoint.resolve(), "cpu")]
    assert payload["players"]["opponent"]["completedTurns"] == 1
    assert payload["legalActions"]


def test_mcts_budget_changes_apply_to_next_search_without_changing_board(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("splendor._native_search")
    checkpoint = tmp_path / "round24.pt"
    checkpoint.write_bytes(b"test checkpoint")
    model = ActorCritic().eval()
    monkeypatch.setattr(web_app, "load_frozen_actor_critic_checkpoint", lambda *a, **kw: model)
    service = WebGameService(
        checkpoint_dir=tmp_path / "play_checkpoints",
        failure_dir=tmp_path / "failures",
        replay_dir=tmp_path / "replays",
        mcts_checkpoint=checkpoint,
        mcts_label="第24轮 · θ53898",
    )
    listed = service.available_opponents()[-1]
    assert listed["searchBudget"] == {"simulations": 1024, "determinizations": 1}
    initial = service.create_game(
        opponent="mcts", human_first=False, seed=43,
        search_budget={"simulations": 4, "determinizations": 1},
    )
    policy = service._games[initial["gameId"]].opponent
    assert isinstance(policy, NativePUCTSearchPolicy)
    assert str(policy.device) == "cpu"
    assert initial["players"]["opponent"]["completedTurns"] == 1
    assert tuple(policy.last_search_result.simulation_budgets) == (4,)

    updated = service.set_search_budget(
        initial["gameId"], {"simulations": 6, "determinizations": 2}
    )
    assert updated["revision"] == initial["revision"]
    assert updated["market"] == initial["market"]
    assert updated["players"] == initial["players"]
    assert updated["searchBudget"] == {"simulations": 6, "determinizations": 2}
    assert "第24轮 · θ53898 · MCTS S6/N2" in updated["opponentLabel"]
    moved = service.apply_action(initial["gameId"], revision=updated["revision"], action_index=0)
    assert tuple(policy.last_search_result.simulation_budgets) == (3, 3)
    assert policy.config.root_noise_fraction == 0.0
    restored = service.undo(initial["gameId"], revision=moved["revision"])
    assert restored["searchBudget"] == {"simulations": 6, "determinizations": 2}
    assert restored["evaluation"] == initial["evaluation"]


@pytest.mark.parametrize("budget", [
    {"simulations": 0, "determinizations": 1},
    {"simulations": 1, "determinizations": 2},
    {"simulations": True, "determinizations": 1},
    {"simulations": 4.5, "determinizations": 1},
    {"simulations": 4},
    "1024",
])
def test_invalid_mcts_budget_is_rejected_before_loading_model(
    tmp_path: Path, budget: object
) -> None:
    checkpoint = tmp_path / "round24.pt"
    checkpoint.write_bytes(b"not a loadable checkpoint")
    service = WebGameService(
        checkpoint_dir=tmp_path / "play_checkpoints", mcts_checkpoint=checkpoint
    )
    with pytest.raises(ValueError):
        service.create_game(
            opponent="mcts", human_first=True, seed=1, search_budget=budget
        )


def test_search_budget_requires_mcts_opponent(tmp_path: Path) -> None:
    service = _service(tmp_path)
    initial = service.create_game(opponent="score", human_first=True, seed=1)
    with pytest.raises(ValueError, match="MCTS"):
        service.set_search_budget(
            initial["gameId"], {"simulations": 4, "determinizations": 1}
        )


def test_web_cli_builds_requested_mcts_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "round24.pt"
    checkpoint.write_bytes(b"test checkpoint")
    services = []

    class FakeServer:
        def __init__(self, address, service):
            services.append(service)

        def serve_forever(self):
            pass

        def server_close(self):
            pass

    monkeypatch.setattr(web_app, "WebGameServer", FakeServer)
    assert web_app.main([
        "--mcts-checkpoint", str(checkpoint), "--mcts-label", "第24轮",
        "--mcts-simulations", "512", "--mcts-determinizations", "2",
        "--mcts-seed", "7",
    ]) == 0
    assert services[0].mcts_checkpoint == checkpoint.resolve()
    assert services[0].mcts_config == MCTSConfig(
        simulations=512, determinizations=2, seed=7, root_noise_fraction=0.0
    )


@pytest.mark.parametrize(("value", "rate"), [(-2.0, 0.0), (-0.6, 0.2), (0.0, 0.5), (0.6, 0.8), (2.0, 1.0)])
def test_web_value_bar_uses_human_request_and_clips_only_display_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: float, rate: float
) -> None:
    service = _service(tmp_path)
    checkpoint = service.checkpoint_dir / "model.pt"
    checkpoint.write_bytes(b"test checkpoint")
    model = ActorCritic().eval()
    with torch.no_grad():
        for parameter in model.critic.parameters():
            parameter.zero_()
        model.critic[-1].bias.fill_(value)
    monkeypatch.setattr(web_app, "load_frozen_actor_critic_checkpoint", lambda *a, **kw: model)
    payload = service.create_game(opponent="checkpoint:model.pt", human_first=False, seed=43)
    assert payload["evaluation"]["source"] == "model"
    assert payload["evaluation"]["value"] == pytest.approx(value)
    assert payload["evaluation"]["humanScoreRate"] == pytest.approx(rate)
    assert payload["evaluation"]["opponentScoreRate"] == pytest.approx(1.0 - rate)
    game = service._games[payload["gameId"]]
    requests = []
    original = game.evaluator.value_only

    def record(request_batch):
        requests.extend(request_batch)
        return original(request_batch)

    monkeypatch.setattr(game.evaluator, "value_only", record)
    web_app._game_payload(payload["gameId"], game)
    assert requests[0].observation == game.engine.observation_for(game.human_player)
    assert requests[0].observation.self_is_first_player is False


def test_undo_restores_rng_replay_history_and_rejects_stale_clicks(tmp_path: Path) -> None:
    service = _service(tmp_path)
    initial = service.create_game(opponent="development", human_first=True, seed=5)
    game_id = initial["gameId"]
    game = service._games[game_id]
    digest = game.engine.current_state_digest
    replay = game.engine.replay_record
    assert initial["canUndo"] is False
    assert initial["evaluation"] is None
    with pytest.raises(ValueError, match="no previous"):
        service.undo(game_id, revision=initial["revision"])
    moved = service.apply_action(game_id, revision=initial["revision"], action_index=0)
    assert moved["canUndo"] is True
    with pytest.raises(ValueError, match="stale"):
        service.undo(game_id, revision=initial["revision"])
    restored = service.undo(game_id, revision=moved["revision"])
    assert restored["revision"] > moved["revision"]
    assert restored["canUndo"] is False
    assert game.engine.current_state_digest == digest
    assert game.engine.replay_record == replay
    for field in ("players", "market", "supply", "legalActions", "phase", "history"):
        assert restored[field] == initial[field]
    with pytest.raises(ValueError, match="stale"):
        service.apply_action(game_id, revision=moved["revision"], action_index=0)
    repeated = service.apply_action(game_id, revision=restored["revision"], action_index=0)
    for field in ("players", "market", "supply", "legalActions", "history"):
        assert repeated[field] == moved[field]


def test_payment_and_finished_game_can_be_undone(tmp_path: Path) -> None:
    service = _service(tmp_path)
    payload = service.create_game(opponent="score", human_first=True, seed=5)
    game_id = payload["gameId"]
    chooser = ScorePolicy()
    checked_payment = False
    for _ in range(300):
        if payload["result"] is not None:
            break
        game = service._games[game_id]
        request = game.response
        before = payload
        action = chooser.choose(request)
        index = request.legal_actions.index(action)
        payload = service.apply_action(game_id, revision=payload["revision"], action_index=index)
        if payload["phase"] == "payment" and not checked_payment:
            payment = payload
            restored = service.undo(game_id, revision=payload["revision"])
            assert restored["phase"] == "main_action"
            assert restored["market"] == before["market"]
            payload = service.apply_action(game_id, revision=restored["revision"], action_index=index)
            assert payload["phase"] == "payment"
            assert payload["legalActions"] == payment["legalActions"]
            checked_payment = True
    assert checked_payment
    assert payload["result"] is not None
    assert payload["evaluation"]["source"] == "result"
    payoff = service._games[game_id].response.payoff(0)
    assert payload["evaluation"]["humanScoreRate"] == (payoff + 1) / 2
    restored = service.undo(game_id, revision=payload["revision"])
    assert restored["result"] is None and restored["legalActions"]
    assert service._games[game_id].engine.replay_record.result is None


def test_web_autosaves_replay_through_terminal_and_undo(tmp_path: Path) -> None:
    from splendor.engine import GameEngine, replay_game

    service = _service(tmp_path)
    payload = service.create_game(opponent="score", human_first=False, seed=100)
    game_id = payload["gameId"]
    path = service.replay_dir / f"{game_id}.json"
    game = service._games[game_id]

    def verify_saved_branch() -> None:
        saved = GameEngine.restore(path, catalog=service.catalog)
        rebuilt = replay_game(saved.replay_record, catalog=service.catalog)
        assert saved.current_state_digest == game.engine.current_state_digest
        assert rebuilt.current_state_digest == saved.current_state_digest
        assert saved.replay_record == game.engine.replay_record

    verify_saved_branch()
    assert game.engine.replay_record.events  # AI-first opening is included.
    chooser = ScorePolicy()
    for _ in range(300):
        if payload["result"] is not None:
            break
        index = game.response.legal_actions.index(chooser.choose(game.response))
        payload = service.apply_action(game_id, revision=payload["revision"], action_index=index)
    assert payload["result"] is not None
    verify_saved_branch()
    assert GameEngine.restore(path).replay_record.result is not None
    payload = service.undo(game_id, revision=payload["revision"])
    verify_saved_branch()
    assert GameEngine.restore(path).replay_record.result is None
    assert list(service.replay_dir.iterdir()) == [path]
