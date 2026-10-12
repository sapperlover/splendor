"""Small local human-vs-policy web application.

The browser receives only an Observation-derived payload and the current legal
action candidates.  All legality and transitions remain owned by GameEngine.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import uuid
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch

from .actor_critic import ActorCritic
from .actions import (
    Action,
    BuyMarket,
    BuyReserved,
    ClaimNoble,
    Pay,
    ReserveDeck,
    ReserveMarket,
    ReturnTokens,
    TakeDistinct,
    TakePair,
)
from .catalog import Catalog, load_default_catalog
from .contracts import (
    CardView,
    DecisionRequest,
    NobleView,
    Observation,
    Policy,
    PublicReservedCard,
)
from .engine import GameEngine, IllegalActionError
from .heuristic_policy import DevelopmentPolicy, ScorePolicy
from .model import Color, DecisionPhase, GemVector, ReservationSource, Tier, TokenVector
from .mcts_policy import MCTSConfig
from .native_policy import NativePUCTSearchPolicy
from .neural_policy import GreedyActorCriticPolicy
from .opponent_pool import load_frozen_actor_critic_checkpoint
from .replay import ActionEvent, ForcedSkipEvent, ReplayEvent
from .result import TerminalResult
from .search_inference import FrozenSearchInference
from .search_policy import PolicyValueSearchPolicy, SearchConfig


_STATIC_ROOT = Path(__file__).with_name("web")
_DEFAULT_CHECKPOINT_ROOT = Path(__file__).resolve().parents[2] / "play_checkpoints"
_COLOR_NAMES = {
    Color.WHITE: "白",
    Color.BLUE: "蓝",
    Color.GREEN: "绿",
    Color.RED: "红",
    Color.BLACK: "黑",
    Color.GOLD: "黄金",
}


@dataclass(slots=True)
class _WebGame:
    engine: GameEngine
    human_player: int
    opponent_id: str
    opponent_label: str
    opponent: Policy
    response: DecisionRequest | TerminalResult
    evaluator: FrozenSearchInference | None = None
    revision: int = 0
    history: list["_HistoryEntry"] = field(default_factory=list)
    undo_stack: list["_UndoState"] = field(default_factory=list)


@dataclass(slots=True)
class _UndoState:
    engine: GameEngine
    response: DecisionRequest
    history_length: int


@dataclass(slots=True)
class _HistoryEntry:
    sequence: int
    actor: int
    kind: str
    summary: str
    details: list[str] = field(default_factory=list)


def _vector(value: GemVector | TokenVector) -> dict[str, int]:
    colors = tuple(Color) if isinstance(value, TokenVector) else tuple(Color)[:-1]
    return {color.value: value.for_color(color) for color in colors}


def _card(card: CardView) -> dict[str, Any]:
    return {
        "id": card.card_id,
        "tier": card.tier.value,
        "bonus": card.bonus.value,
        "points": card.points,
        "cost": _vector(card.cost),
    }


def _noble(noble: NobleView) -> dict[str, Any]:
    return {
        "id": noble.noble_id,
        "points": noble.points,
        "requirement": _vector(noble.requirement),
    }


def _catalog_card(catalog: Catalog, card_id: str) -> dict[str, Any]:
    definition = catalog.card(card_id)
    return {
        "id": definition.card_id,
        "tier": definition.tier.value,
        "bonus": definition.bonus.value,
        "points": definition.points,
        "cost": _vector(definition.cost),
    }


def _catalog_noble(catalog: Catalog, noble_id: str) -> dict[str, Any]:
    definition = catalog.noble(noble_id)
    return {
        "id": definition.noble_id,
        "points": definition.points,
        "requirement": _vector(definition.requirement),
    }


def _own_reservations(observation: Observation) -> list[dict[str, Any]]:
    blind_by_tier: dict[Tier, list[CardView]] = {tier: [] for tier in Tier}
    for card in observation.self_private.blind_reserved_cards:
        blind_by_tier[card.tier].append(card)
    for cards in blind_by_tier.values():
        cards.sort(key=lambda item: item.card_id)

    result: list[dict[str, Any]] = []
    for reserved in observation.self_public.reserved_cards:
        if reserved.card is not None:
            result.append({"source": reserved.source.value, "hidden": False, "card": _card(reserved.card)})
            continue
        candidates = blind_by_tier[reserved.tier]
        if not candidates:
            raise AssertionError("private blind reservation does not match public reservation")
        result.append({"source": reserved.source.value, "hidden": False, "card": _card(candidates.pop(0))})
    return result


def _opponent_reservations(
    reservations: tuple[PublicReservedCard, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "source": reserved.source.value,
            "hidden": reserved.card is None,
            "tier": reserved.tier.value,
            "card": None if reserved.card is None else _card(reserved.card),
        }
        for reserved in reservations
    ]


def _player_payload(
    observation: Observation,
    catalog: Catalog,
    *,
    own: bool,
) -> dict[str, Any]:
    public = observation.self_public if own else observation.opponent_public
    purchased = [_catalog_card(catalog, card_id) for card_id in public.purchased_card_ids]
    purchased.sort(key=lambda card: (card["bonus"], card["tier"], card["points"], card["id"]))
    return {
        "isSelf": own,
        "isFirst": observation.self_is_first_player if own else not observation.self_is_first_player,
        "score": public.score,
        "tokens": _vector(public.tokens),
        "bonuses": _vector(public.bonuses),
        "purchased": purchased,
        "nobles": [_catalog_noble(catalog, noble_id) for noble_id in public.noble_ids],
        "reserved": (
            _own_reservations(observation)
            if own
            else _opponent_reservations(public.reserved_cards)
        ),
        "completedTurns": (
            observation.own_completed_turns if own else observation.opponent_completed_turns
        ),
    }


def _token_text(tokens: TokenVector) -> str:
    parts = [
        f"{_COLOR_NAMES[color]}×{tokens.for_color(color)}"
        for color in Color
        if tokens.for_color(color)
    ]
    return "、".join(parts) if parts else "0"


def _card_history_text(catalog: Catalog, card_id: str) -> str:
    card = catalog.card(card_id)
    return (
        f"{card.tier.value} 级·{_COLOR_NAMES[card.bonus]}色加成·"
        f"{card.points} 分"
    )


def _action_history_text(action: Action, catalog: Catalog) -> str:
    if isinstance(action, TakeDistinct):
        return "拿取 " + "、".join(_COLOR_NAMES[color] for color in action.colors)
    if isinstance(action, TakePair):
        return f"拿取 2 枚{_COLOR_NAMES[action.color]}宝石"
    if isinstance(action, ReserveMarket):
        return "保留市场牌：" + _card_history_text(catalog, action.card_id)
    if isinstance(action, ReserveDeck):
        return f"从 {action.tier.value} 级牌堆盲保留"
    if isinstance(action, BuyMarket):
        return "购买市场牌：" + _card_history_text(catalog, action.card_id)
    if isinstance(action, BuyReserved):
        return "购买保留牌：" + _card_history_text(catalog, action.card_id)
    if isinstance(action, Pay):
        return "支付 " + _token_text(action.tokens)
    if isinstance(action, ReturnTokens):
        return "归还 " + _token_text(action.tokens)
    if isinstance(action, ClaimNoble):
        noble = catalog.noble(action.noble_id)
        return f"获得贵族：{noble.points} 分"
    raise TypeError(f"unsupported history action: {action!r}")


def _record_replay_events(
    game: _WebGame,
    events: tuple[ReplayEvent, ...],
    catalog: Catalog,
) -> None:
    for event in events:
        if isinstance(event, ForcedSkipEvent):
            game.history.append(
                _HistoryEntry(
                    sequence=event.sequence,
                    actor=event.actor,
                    kind="ForcedSkip",
                    summary="无合法主行动，被迫跳过",
                )
            )
            continue
        if not isinstance(event, ActionEvent):  # pragma: no cover - closed union
            raise TypeError(f"unsupported replay event: {event!r}")
        text = _action_history_text(event.action, catalog)
        if event.phase is DecisionPhase.MAIN_ACTION:
            game.history.append(
                _HistoryEntry(
                    sequence=event.sequence,
                    actor=event.actor,
                    kind=type(event.action).__name__,
                    summary=text,
                )
            )
        elif game.history and game.history[-1].actor == event.actor:
            game.history[-1].details.append(text)
        else:
            game.history.append(
                _HistoryEntry(
                    sequence=event.sequence,
                    actor=event.actor,
                    kind=type(event.action).__name__,
                    summary=text,
                )
            )


def _action_payload(index: int, action: Action) -> dict[str, Any]:
    payload: dict[str, Any] = {"index": index, "kind": type(action).__name__}
    if isinstance(action, TakeDistinct):
        colors = [color.value for color in action.colors]
        payload.update(colors=colors, label="拿取 " + " + ".join(_COLOR_NAMES[color] for color in action.colors))
    elif isinstance(action, TakePair):
        payload.update(color=action.color.value, label=f"拿取 2 枚{_COLOR_NAMES[action.color]}宝石")
    elif isinstance(action, ReserveMarket):
        payload.update(cardId=action.card_id, label="保留这张市场牌")
    elif isinstance(action, ReserveDeck):
        payload.update(tier=action.tier.value, label=f"盲保留 {action.tier.value} 级牌")
    elif isinstance(action, BuyMarket):
        payload.update(cardId=action.card_id, label="购买这张市场牌")
    elif isinstance(action, BuyReserved):
        payload.update(cardId=action.card_id, label="购买这张保留牌")
    elif isinstance(action, Pay):
        payload.update(tokens=_vector(action.tokens), label=f"支付：{_token_text(action.tokens)}")
    elif isinstance(action, ReturnTokens):
        payload.update(tokens=_vector(action.tokens), label=f"归还：{_token_text(action.tokens)}")
    elif isinstance(action, ClaimNoble):
        payload.update(nobleId=action.noble_id, label="选择这位贵族")
    else:  # pragma: no cover - Action is a closed union
        raise TypeError(f"unsupported action: {action!r}")
    return payload


def _result_payload(result: TerminalResult, human_player: int) -> dict[str, Any]:
    payoff = result.payoff(human_player)
    return {
        "scores": list(result.scores),
        "purchasedCardCounts": list(result.purchased_card_counts),
        "winnerIds": list(result.winner_ids),
        "completedTurns": list(result.completed_turns),
        "reason": result.reason.value,
        "outcome": "win" if payoff > 0 else "loss" if payoff < 0 else "draw",
        "humanScore": result.scores[human_player],
        "opponentScore": result.scores[1 - human_player],
    }


def _evaluation_payload(game: _WebGame) -> dict[str, Any] | None:
    if isinstance(game.response, TerminalResult):
        value = float(game.response.payoff(game.human_player))
        source = "result"
    elif game.evaluator is not None:
        # The displayed request belongs to the human, including payment/return phases.
        value = game.evaluator.value_only((game.response,))[0]
        source = "model"
    else:
        return None
    human_rate = (max(-1.0, min(1.0, value)) + 1.0) / 2.0
    return {
        "value": value,
        "humanScoreRate": human_rate,
        "opponentScoreRate": 1.0 - human_rate,
        "source": source,
        "perspective": "human",
    }


def _game_payload(game_id: str, game: _WebGame) -> dict[str, Any]:
    if isinstance(game.response, DecisionRequest):
        observation = game.response.observation
        legal_actions = [
            _action_payload(index, action)
            for index, action in enumerate(game.response.legal_actions)
        ]
        result = None
    else:
        observation = game.engine.observation_for(game.human_player)
        legal_actions = []
        result = _result_payload(game.response, game.human_player)

    market = {
        str(tier.value): [_card(card) for card in getattr(observation.market, f"tier_{('one', 'two', 'three')[tier.value - 1]}")]
        for tier in Tier
    }
    return {
        "gameId": game_id,
        "revision": game.revision,
        "canUndo": bool(game.undo_stack),
        "evaluation": _evaluation_payload(game),
        "opponent": game.opponent_id,
        "opponentLabel": game.opponent_label,
        "searchBudget": {
            "simulations": game.opponent.config.simulations,
            "determinizations": game.opponent.config.determinizations,
        } if isinstance(game.opponent, NativePUCTSearchPolicy) else None,
        "phase": observation.phase.value,
        "supply": _vector(observation.supply),
        "deckCounts": {str(tier.value): observation.deck_counts[tier.value - 1] for tier in Tier},
        "market": market,
        "nobles": [_noble(noble) for noble in observation.nobles_in_play],
        "players": {
            "self": _player_payload(observation, game.engine.catalog, own=True),
            "opponent": _player_payload(observation, game.engine.catalog, own=False),
        },
        "pending": None if observation.pending is None else {
            "tokensToReturn": observation.pending.tokens_to_return,
            "paymentTarget": None if observation.pending.payment_target is None else _card(observation.pending.payment_target),
            "eligibleNobles": [_noble(noble) for noble in observation.pending.eligible_nobles],
        },
        "finalRoundTriggered": observation.final_round_triggered,
        "history": [
            {
                "sequence": entry.sequence,
                "actor": "self" if entry.actor == game.human_player else "opponent",
                "kind": entry.kind,
                "summary": entry.summary,
                "details": list(entry.details),
            }
            for entry in reversed(game.history[-8:])
        ],
        "legalActions": legal_actions,
        "result": result,
    }


class WebGameService:
    """In-memory sessions for a local, single-user oriented web UI."""

    def __init__(
        self,
        *,
        failure_dir: Path | None = None,
        checkpoint_dir: Path | None = None,
        search_checkpoint: Path | None = None,
        search_config: SearchConfig | None = None,
        mcts_checkpoint: Path | None = None,
        mcts_config: MCTSConfig | None = None,
        mcts_label: str | None = None,
    ) -> None:
        self.catalog = load_default_catalog()
        self.failure_dir = failure_dir or Path.cwd() / "web_failures"
        self.checkpoint_dir = (checkpoint_dir or _DEFAULT_CHECKPOINT_ROOT).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if search_checkpoint is None and search_config is not None:
            raise ValueError("search_config requires search_checkpoint")
        self.search_checkpoint = (
            None if search_checkpoint is None else search_checkpoint.resolve()
        )
        if self.search_checkpoint is not None:
            if (
                self.search_checkpoint.suffix != ".pt"
                or not self.search_checkpoint.is_file()
            ):
                raise ValueError("search_checkpoint must be an existing .pt file")
            self.search_config = search_config or SearchConfig()
        else:
            self.search_config = None
        if mcts_checkpoint is None and (mcts_config is not None or mcts_label is not None):
            raise ValueError("mcts_config and mcts_label require mcts_checkpoint")
        self.mcts_checkpoint = None if mcts_checkpoint is None else mcts_checkpoint.resolve()
        if self.mcts_checkpoint is not None and (
            self.mcts_checkpoint.suffix != ".pt" or not self.mcts_checkpoint.is_file()
        ):
            raise ValueError("mcts_checkpoint must be an existing .pt file")
        self.mcts_config = mcts_config or MCTSConfig(
            simulations=1024, determinizations=1, root_noise_fraction=0.0
        )
        self.mcts_label = mcts_label or (
            "MCTS" if self.mcts_checkpoint is None else self.mcts_checkpoint.stem
        )
        self._checkpoint_models: dict[tuple[Path, int, int], ActorCritic] = {}
        self._games: dict[str, _WebGame] = {}
        self._lock = threading.RLock()

    def available_opponents(self) -> list[dict[str, Any]]:
        opponents = [
            {"id": "score", "label": "ScorePolicy · 积分优先", "kind": "heuristic"},
            {
                "id": "development",
                "label": "DevelopmentPolicy · 发展优先",
                "kind": "heuristic",
            },
        ]
        opponents.extend(
            {
                "id": f"checkpoint:{path.name}",
                "label": f"PPO · {path.stem} · greedy CPU",
                "kind": "checkpoint",
            }
            for path in sorted(self.checkpoint_dir.glob("*.pt"), key=lambda item: item.name)
            if path.is_file()
        )
        if self.search_checkpoint is not None and self.search_config is not None:
            config = self.search_config
            fold_label = " · 折叠强制阶段" if config.fold_forced_phases else ""
            opponents.append(
                {
                    "id": "search",
                    "label": (
                        f"PPO 搜索 · {self.search_checkpoint.stem} · "
                        f"D{config.depth} top{config.top_k} · "
                        f"{config.determinizations} determinization{fold_label}"
                    ),
                    "kind": "search_checkpoint",
                }
            )
        if self.mcts_checkpoint is not None:
            opponents.append({
                "id": "mcts",
                "label": self._mcts_opponent_label(self.mcts_config),
                "kind": "mcts_checkpoint",
                "searchBudget": {
                    "simulations": self.mcts_config.simulations,
                    "determinizations": self.mcts_config.determinizations,
                },
            })
        return opponents

    def _mcts_opponent_label(self, config: MCTSConfig) -> str:
        return (
            f"{self.mcts_label} · MCTS S{config.simulations}/N{config.determinizations} · CPU"
        )

    @staticmethod
    def _with_budget(config: MCTSConfig, budget: dict[str, Any]) -> MCTSConfig:
        if not isinstance(budget, dict) or set(budget) != {"simulations", "determinizations"}:
            raise ValueError("searchBudget must contain simulations and determinizations")
        return replace(config, **budget)

    def _load_checkpoint(self, path: Path) -> ActorCritic:
        stat = path.stat()
        cache_key = (path, stat.st_mtime_ns, stat.st_size)
        model = self._checkpoint_models.get(cache_key)
        if model is None:
            self._checkpoint_models = {
                key: cached
                for key, cached in self._checkpoint_models.items()
                if key[0] != path
            }
            model = load_frozen_actor_critic_checkpoint(path, device="cpu")
            self._checkpoint_models[cache_key] = model
        return model

    def _opponent_policy(
        self, opponent_id: str, search_budget: dict[str, Any] | None = None
    ) -> tuple[str, Policy]:
        if opponent_id == "mcts":
            if self.mcts_checkpoint is None:
                raise ValueError("MCTS opponent is not configured")
            config = self.mcts_config
            if search_budget is not None:
                config = self._with_budget(config, search_budget)
            model = self._load_checkpoint(self.mcts_checkpoint)
            return self._mcts_opponent_label(config), NativePUCTSearchPolicy(
                model, config=config, device="cpu", catalog=self.catalog
            )
        if search_budget is not None:
            raise ValueError("searchBudget requires an MCTS opponent")
        if opponent_id == "score":
            return "ScorePolicy", ScorePolicy()
        if opponent_id == "development":
            return "DevelopmentPolicy", DevelopmentPolicy()
        if opponent_id == "search":
            if self.search_checkpoint is None or self.search_config is None:
                raise ValueError("search opponent is not configured")
            model = self._load_checkpoint(self.search_checkpoint)
            config = self.search_config
            label = (
                f"PPO 搜索 · {self.search_checkpoint.stem} · "
                f"D{config.depth} top{config.top_k}"
            )
            return label, PolicyValueSearchPolicy(
                model,
                config=config,
                device="cpu",
                catalog=self.catalog,
            )
        prefix = "checkpoint:"
        if not opponent_id.startswith(prefix):
            raise ValueError("unknown opponent")
        filename = opponent_id[len(prefix):]
        if not filename or Path(filename).name != filename or not filename.endswith(".pt"):
            raise ValueError("invalid playable checkpoint id")
        path = (self.checkpoint_dir / filename).resolve()
        if path.parent != self.checkpoint_dir or not path.is_file():
            raise ValueError("playable checkpoint does not exist")
        model = self._load_checkpoint(path)
        label = f"PPO · {path.stem}"
        return label, GreedyActorCriticPolicy(model, device="cpu")

    def create_game(
        self, *, opponent: str, human_first: bool, seed: int,
        search_budget: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(opponent, str):
            raise ValueError("opponent must be a string id")
        if not isinstance(human_first, bool):
            raise ValueError("humanFirst must be a boolean")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        opponent_label, opponent_policy = self._opponent_policy(opponent, search_budget)
        human_player = 0
        engine = GameEngine.new(
            seed,
            first_player=human_player if human_first else 1,
            catalog=self.catalog,
            failure_dir=self.failure_dir,
        )
        replay_start = len(engine.replay_record.events)
        response = engine.decision()
        evaluator = None
        if isinstance(opponent_policy, (NativePUCTSearchPolicy, PolicyValueSearchPolicy)):
            evaluator = opponent_policy.evaluator
        elif isinstance(opponent_policy, GreedyActorCriticPolicy):
            evaluator = FrozenSearchInference(opponent_policy.model, device="cpu")
        game = _WebGame(
            engine=engine,
            human_player=human_player,
            opponent_id=opponent,
            opponent_label=opponent_label,
            opponent=opponent_policy,
            response=response,
            evaluator=evaluator,
        )
        _record_replay_events(
            game, engine.replay_record.events[replay_start:], self.catalog
        )
        self._advance_opponent(game)
        game_id = uuid.uuid4().hex
        with self._lock:
            self._games[game_id] = game
        return _game_payload(game_id, game)

    def set_search_budget(self, game_id: str, budget: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            game = self._games.get(game_id)
            if game is None:
                raise KeyError("unknown game")
            if not isinstance(game.opponent, NativePUCTSearchPolicy):
                raise ValueError("searchBudget requires an MCTS opponent")
            config = self._with_budget(game.opponent.config, budget)
            game.opponent.config = config
            game.opponent_label = self._mcts_opponent_label(config)
            return _game_payload(game_id, game)

    def apply_action(self, game_id: str, *, revision: int, action_index: int) -> dict[str, Any]:
        with self._lock:
            game = self._games.get(game_id)
            if game is None:
                raise KeyError("unknown game")
            if isinstance(revision, bool) or not isinstance(revision, int):
                raise ValueError("revision must be an integer")
            if revision != game.revision:
                raise ValueError("stale game revision; refresh the displayed state")
            if not isinstance(game.response, DecisionRequest):
                raise ValueError("the game is already over")
            if game.engine.current_player != game.human_player:
                raise AssertionError("human action requested during opponent turn")
            if isinstance(action_index, bool) or not isinstance(action_index, int):
                raise ValueError("actionIndex must be an integer")
            if action_index < 0 or action_index >= len(game.response.legal_actions):
                raise ValueError("actionIndex is not a current legal candidate")
            action = game.response.legal_actions[action_index]
            game.undo_stack.append(_UndoState(
                engine=game.engine.fork(),
                response=game.response,
                history_length=len(game.history),
            ))
            game.response = self._apply_and_record(game, action)
            game.revision += 1
            self._advance_opponent(game)
            return _game_payload(game_id, game)

    def undo(self, game_id: str, *, revision: int) -> dict[str, Any]:
        with self._lock:
            game = self._games.get(game_id)
            if game is None:
                raise KeyError("unknown game")
            if isinstance(revision, bool) or not isinstance(revision, int):
                raise ValueError("revision must be an integer")
            if revision != game.revision:
                raise ValueError("stale game revision; refresh the displayed state")
            if not game.undo_stack:
                raise ValueError("no previous human action to undo")
            previous = game.undo_stack.pop()
            game.engine = previous.engine
            game.response = previous.response
            del game.history[previous.history_length:]
            # Never reuse an old revision: a delayed pre-undo click must remain stale.
            game.revision += 1
            return _game_payload(game_id, game)

    def _apply_and_record(
        self, game: _WebGame, action: Action
    ) -> DecisionRequest | TerminalResult:
        replay_start = len(game.engine.replay_record.events)
        response = game.engine.apply(action)
        _record_replay_events(
            game,
            game.engine.replay_record.events[replay_start:],
            self.catalog,
        )
        return response

    def _advance_opponent(self, game: _WebGame) -> None:
        while (
            isinstance(game.response, DecisionRequest)
            and game.engine.current_player != game.human_player
        ):
            action = game.opponent.choose(game.response)
            game.response = self._apply_and_record(game, action)
            game.revision += 1


class _Handler(BaseHTTPRequestHandler):
    server: "WebGameServer"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path == "/api/opponents":
            self._send_json(
                HTTPStatus.OK,
                {"opponents": self.server.service.available_opponents()},
            )
            return
        assets = {"/": "index.html", "/app.js": "app.js", "/styles.css": "styles.css"}
        asset = assets.get(path)
        if asset is None:
            self._json_error(HTTPStatus.NOT_FOUND, "not_found", "resource not found")
            return
        data = (_STATIC_ROOT / asset).read_bytes()
        content_type = mimetypes.guess_type(asset)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            body = self._read_json()
            path = urlparse(self.path).path
            if path == "/api/games":
                payload = self.server.service.create_game(
                    opponent=body.get("opponent"),
                    human_first=body.get("humanFirst"),
                    seed=body.get("seed"),
                    search_budget=body.get("searchBudget"),
                )
                self._send_json(HTTPStatus.CREATED, payload)
                return
            pieces = path.strip("/").split("/")
            if len(pieces) == 4 and pieces[:2] == ["api", "games"] and pieces[3] == "undo":
                payload = self.server.service.undo(pieces[2], revision=body.get("revision"))
                self._send_json(HTTPStatus.OK, payload)
                return
            if len(pieces) == 4 and pieces[:2] == ["api", "games"] and pieces[3] == "search-budget":
                payload = self.server.service.set_search_budget(
                    pieces[2], body.get("searchBudget")
                )
                self._send_json(HTTPStatus.OK, payload)
                return
            if len(pieces) == 4 and pieces[:2] == ["api", "games"] and pieces[3] == "actions":
                payload = self.server.service.apply_action(
                    pieces[2],
                    revision=body.get("revision"),
                    action_index=body.get("actionIndex"),
                )
                self._send_json(HTTPStatus.OK, payload)
                return
            self._json_error(HTTPStatus.NOT_FOUND, "not_found", "resource not found")
        except (ValueError, KeyError, IllegalActionError) as error:
            self._json_error(HTTPStatus.BAD_REQUEST, "bad_request", str(error))
        except json.JSONDecodeError:
            self._json_error(HTTPStatus.BAD_REQUEST, "invalid_json", "request body must be JSON")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 64 * 1024:
            raise ValueError("request body size is invalid")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json_error(self, status: HTTPStatus, code: str, message: str) -> None:
        self._send_json(status, {"error": code, "message": message})

    def log_message(self, format: str, *args: object) -> None:
        print(f"[web] {self.address_string()} {format % args}")


class WebGameServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], service: WebGameService) -> None:
        self.service = service
        super().__init__(address, _Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local Splendor human-vs-policy web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint-dir", type=Path, default=_DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--mcts-checkpoint", type=Path, help="add a native MCTS opponent with adjustable S/N budgets")
    parser.add_argument("--mcts-label", help="display name for the MCTS checkpoint")
    parser.add_argument("--mcts-simulations", type=int, default=1024)
    parser.add_argument("--mcts-determinizations", type=int, default=1)
    parser.add_argument("--mcts-seed", type=int, default=0)
    parser.add_argument(
        "--search-checkpoint",
        type=Path,
        help="add one top-k search opponent backed by this checkpoint",
    )
    parser.add_argument("--search-depth", type=int, default=4)
    parser.add_argument("--search-top-k", type=int, default=5)
    parser.add_argument("--search-determinizations", type=int, default=1)
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--search-leaf-value-clip", type=float, default=1.0)
    parser.add_argument(
        "--search-fold-forced-phases",
        action="store_true",
        help="collapse forced payment/return/noble decisions in the search tree",
    )
    args = parser.parse_args(argv)

    mcts_config = None
    if args.mcts_checkpoint is None:
        if (args.mcts_label is not None or args.mcts_simulations != 1024
                or args.mcts_determinizations != 1 or args.mcts_seed != 0):
            parser.error("MCTS options require --mcts-checkpoint")
    else:
        try:
            mcts_config = MCTSConfig(
                simulations=args.mcts_simulations,
                determinizations=args.mcts_determinizations,
                seed=args.mcts_seed,
                root_noise_fraction=0.0,
            )
        except ValueError as error:
            parser.error(str(error))

    search_config = None
    if args.search_checkpoint is None:
        configured_search_options = (
            args.search_depth != 4
            or args.search_top_k != 5
            or args.search_determinizations != 1
            or args.search_seed != 0
            or args.search_leaf_value_clip != 1.0
            or args.search_fold_forced_phases
        )
        if configured_search_options:
            parser.error("search options require --search-checkpoint")
    else:
        try:
            search_config = SearchConfig(
                top_k=args.search_top_k,
                depth=args.search_depth,
                determinizations=args.search_determinizations,
                seed=args.search_seed,
                leaf_value_clip=args.search_leaf_value_clip,
                fold_forced_phases=args.search_fold_forced_phases,
            )
        except ValueError as error:
            parser.error(str(error))

    # Keep single-match inference to one CPU thread alongside training.
    torch.set_num_threads(1)
    server = WebGameServer(
        (args.host, args.port),
        WebGameService(
            checkpoint_dir=args.checkpoint_dir,
            search_checkpoint=args.search_checkpoint,
            search_config=search_config,
            mcts_checkpoint=args.mcts_checkpoint,
            mcts_config=mcts_config,
            mcts_label=args.mcts_label,
        ),
    )
    print(f"Splendor web UI: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
