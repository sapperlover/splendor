from __future__ import annotations

from pathlib import Path

import pytest

from splendor.actor_critic import ActorCritic
import splendor.web_app as web_app
from splendor.contracts import PublicReservedCard
from splendor.actions import ReserveDeck
from splendor.model import ReservationSource, Tier
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

        def choose(self, request: object) -> object:
            return request.legal_actions[0]  # type: ignore[attr-defined]

    monkeypatch.setattr(web_app, "load_frozen_actor_critic_checkpoint", load_model)
    monkeypatch.setattr(web_app, "PolicyValueSearchPolicy", FakeSearchPolicy)
    service = WebGameService(
        failure_dir=tmp_path / "failures",
        checkpoint_dir=tmp_path / "play_checkpoints",
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
