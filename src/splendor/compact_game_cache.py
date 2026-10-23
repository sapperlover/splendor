"""Lossless per-game feature dictionaries; decode only sampled rows/slices.

Training on the 19 GiB decoded pool otherwise repeatedly opens ZIP members.
These float32 dictionaries preserve every input bit, including signed zero.
The original archives remain the source of truth; this is execution storage.
"""
from __future__ import annotations

import time

import numpy as np

from .alphazero_replay import _GameCache
from .selfplay_data import STATE_FIELDS, SelfPlaySample, load_game

NEEDED = (*STATE_FIELDS, "candidate_offsets", "candidate_features", "policy_target",
          "behavior_policy", "root_network_prior", "value_target", "policy_weight", "value_weight", "decision_index", "player_id",
          "completed_player_turns")


class DictionaryArray:
    def __init__(self, array):
        # Use bit patterns rather than float equality: -0 and +0 must remain
        # distinct. NaNs are already forbidden by the source data contract.
        bits, inverse = np.unique(array.view(np.uint32), return_inverse=True)
        if len(bits) > 65536:
            raise ValueError("feature dictionary exceeds uint16 capacity")
        self.values = bits.view(np.float32)
        self.codes = inverse.astype(np.uint8 if len(bits) <= 256 else np.uint16).reshape(array.shape)
        self.nbytes = self.values.nbytes + self.codes.nbytes

    def __getitem__(self, key):
        return self.values[self.codes[key]]


def compact(arrays, metadata):
    result = {name: DictionaryArray(arrays[name]) if arrays[name].dtype == np.float32
              and (name in STATE_FIELDS or name == "candidate_features") else arrays[name]
              for name in NEEDED}
    size = sum(a.nbytes for a in result.values())
    meta = {k: metadata[k] for k in ("game_id", "run_id", "round_index", "split", "terminal_result")}
    return result, meta, size


class CompactGameCache(_GameCache):
    def __init__(self, max_bytes, *, progress_targets=False):
        super().__init__(max_bytes)
        self.progress_targets = progress_targets
        self.encode_seconds = 0.
        self.raw_decoded_bytes = 0

    def stats(self):
        return {**super().stats(), "storage":"lossless-float32-dictionary-v1",
                "encode_seconds":self.encode_seconds,"raw_decoded_bytes":self.raw_decoded_bytes}

    def sample(self, game, decision_index):
        key = game.key
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            tick = time.perf_counter()
            arrays, metadata = load_game(game.path)
            self.load_seconds += time.perf_counter()-tick
            self.decode_count += 1
            self.raw_decoded_bytes += sum(a.nbytes for a in arrays.values())
            tick = time.perf_counter()
            entry = compact(arrays, metadata)
            if self.progress_targets:
                from .progress_auxiliary import game_progress_targets
                extra = game_progress_targets(arrays, metadata)
                entry = (entry[0] | extra, entry[1], entry[2] + sum(a.nbytes for a in extra.values()))
            self.encode_seconds += time.perf_counter()-tick
            size = entry[2]
            self.decoded_bytes += size
            if size <= self.max_bytes:
                self._entries[key] = entry
                self._bytes += size
                while self._bytes > self.max_bytes:
                    _, evicted = self._entries.popitem(last=False)
                    self._bytes -= evicted[2]
                self.peak_bytes = max(self.peak_bytes,self._bytes)
                self.peak_entries = max(self.peak_entries,len(self._entries))
        else:
            self.hits += 1
            self._entries.move_to_end(key)
        arrays, metadata, _ = entry
        return SelfPlaySample(arrays, metadata, decision_index)
