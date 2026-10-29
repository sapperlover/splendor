"""Ordered CPU batch preparation over one immutable anonymous shared arena.

Workers use spawn and read-only mappings of a Linux memfd. Compact payloads live
in the shared mapping; workers receive small array descriptors and keep bounded
local metadata caches. They never own a replay LRU or sample a random position.
"""
from __future__ import annotations

from collections import OrderedDict, deque
import ctypes
import gc
import mmap
import multiprocessing as mp
import os
import pickle
import time

import numpy as np

from .compact_game_cache import DictionaryArray, compact
from .continuation_auxiliary import ContinuationAuxiliaryConfig, continuation_targets, game_continuation_targets
from .rng import DeterministicRng
from .selfplay_data import SelfPlaySample, collate_selfplay_samples, load_game

_worker_arena = None
_worker_auxiliary = None
_worker_entries = None


def _anonymous_file(capacity):
    # This Conda Python omits os.memfd_create; glibc on the target Linux exposes it.
    library = ctypes.CDLL(None, use_errno=True)
    create = library.memfd_create
    create.argtypes, create.restype = (ctypes.c_char_p, ctypes.c_uint), ctypes.c_int
    descriptor = create(b"splendor-training-cache", 1)  # MFD_CLOEXEC
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    os.ftruncate(descriptor, capacity)
    return descriptor


def _load_game(job):
    game, auxiliary = job
    tick = time.perf_counter()
    arrays, metadata = load_game(game.path)
    load_seconds = time.perf_counter() - tick
    raw_bytes = sum(a.nbytes for a in arrays.values())
    tick = time.perf_counter()
    extra = game_continuation_targets(arrays, metadata, auxiliary)
    entry, metadata, size = compact(arrays, metadata)
    entry.update(extra)
    return game.key, entry, metadata, size + sum(a.nbytes for a in extra.values()), raw_bytes, load_seconds, time.perf_counter()-tick


def _worker_init(arena_path, capacity, auxiliary):
    global _worker_arena, _worker_auxiliary, _worker_entries
    descriptor = os.open(arena_path, os.O_RDONLY)
    try:
        _worker_arena = mmap.mmap(descriptor, capacity, access=mmap.ACCESS_READ)
    finally:
        os.close(descriptor)
    _worker_auxiliary = auxiliary
    _worker_entries = OrderedDict()
    # The worker owns only bounded descriptors/batches, never a replay LRU.
    gc.disable()


def _array_view(spec):
    offset, shape, dtype = spec
    array = np.ndarray(shape, dtype=np.dtype(dtype), buffer=_worker_arena, offset=offset)
    array.flags.writeable = False
    return array


def _sample(specification, decision_index):
    entry = _worker_entries.get(specification)
    if entry is None:
        specs, metadata = pickle.loads(specification)
        arrays = {}
        for name, spec in specs.items():
            if spec[0] == "dictionary":
                value = DictionaryArray.__new__(DictionaryArray)
                value.values, value.codes = _array_view(spec[1]), _array_view(spec[2])
                value.nbytes = value.values.nbytes + value.codes.nbytes
            else:
                value = _array_view(spec[1])
            arrays[name] = value
        entry = arrays, metadata
        _worker_entries[specification] = entry
        if len(_worker_entries) > 1024:
            _worker_entries.popitem(last=False)
    else:
        _worker_entries.move_to_end(specification)
    return SelfPlaySample(*entry, decision_index)


def _prepare_batch(job):
    policy_specs, value_specs = job
    tick = time.perf_counter()
    policy_samples = [_sample(*spec) for spec in policy_specs]
    value_samples = [_sample(*spec) for spec in value_specs]
    policy = collate_selfplay_samples(policy_samples)
    value = collate_selfplay_samples(value_samples)
    labels = continuation_targets(value_samples, _worker_auxiliary)
    return policy, value, labels, time.perf_counter()-tick, os.getpid()


class SharedReadonlyPrefetch:
    """One shared training pool, bounded ordered batches, committed RNG only."""

    def __init__(self, index, drawer, workers, *, progress=None):
        if workers <= 0 or index.cache.continuation_auxiliary is None:
            raise ValueError("shared prefetch requires workers and continuation targets")
        self.cache, self.drawer, self.workers = index.cache, drawer, workers
        self.auxiliary = self.cache.continuation_auxiliary
        self.descriptor = _anonymous_file(self.cache.max_bytes)
        self.arena = mmap.mmap(self.descriptor, self.cache.max_bytes)
        self.used_bytes = 0
        self.specifications = {}
        self.pending = deque()
        self.pool = None
        self.closed = False
        self.remaining = 0
        games = index.split_games("train")
        if self.cache.entries:
            raise ValueError("shared training data must be built before sampling")
        started = time.perf_counter()
        try:
            with mp.get_context("spawn").Pool(workers) as loaders:
                jobs = ((game, self.auxiliary) for game in games)
                for number, loaded in enumerate(loaders.imap(_load_game, jobs, chunksize=1), 1):
                    self._store(loaded)
                    if progress and (number % 1000 == 0 or number == len(games)):
                        progress({"games": number, "total_games": len(games), "payload_bytes": self.cache.bytes,
                                  "arena_bytes": self.used_bytes, "seconds": time.perf_counter()-started})
            arena_path = f"/proc/{os.getpid()}/fd/{self.descriptor}"
            self.pool = mp.get_context("spawn").Pool(workers, initializer=_worker_init,
                initargs=(arena_path, self.cache.max_bytes, self.auxiliary))
        except BaseException:
            self.close()
            raise
        self.warmup_seconds = time.perf_counter()-started

    def _store_array(self, source):
        offset = (self.used_bytes + 63) // 64 * 64
        end = offset + source.nbytes
        if end > self.cache.max_bytes:
            raise ValueError("immutable training pool exceeds shared cache capacity")
        target = np.ndarray(source.shape, dtype=source.dtype, buffer=self.arena, offset=offset)
        target[...] = source
        target.flags.writeable = False
        self.used_bytes = end
        return target, (offset, source.shape, source.dtype.str)

    def _store(self, loaded):
        key, arrays, metadata, size, raw_bytes, load_seconds, encode_seconds = loaded
        stored, specs = {}, {}
        for name, source in arrays.items():
            if isinstance(source, DictionaryArray):
                value = DictionaryArray.__new__(DictionaryArray)
                value.values, values_spec = self._store_array(source.values)
                value.codes, codes_spec = self._store_array(source.codes)
                value.nbytes = source.nbytes
                specs[name] = ("dictionary", values_spec, codes_spec)
            else:
                value, spec = self._store_array(source)
                specs[name] = ("array", spec)
            stored[name] = value
        self.specifications[key] = pickle.dumps((specs, metadata), protocol=5)
        self.cache._entries[key] = (stored, metadata, size)
        self.cache._bytes += size
        self.cache.misses += 1
        self.cache.decode_count += 1
        self.cache.decoded_bytes += size
        self.cache.raw_decoded_bytes += raw_bytes
        self.cache.load_seconds += load_seconds
        self.cache.encode_seconds += encode_seconds
        self.cache.peak_bytes = self.cache._bytes
        self.cache.peak_entries = self.cache.entries

    def start(self, rng, steps):
        self.rng = DeterministicRng.restore(rng.snapshot())
        self.remaining = steps
        for _ in range(min(self.workers, steps)):
            self._submit()

    def _submit(self):
        drawn = self.drawer(self.rng)
        def specs(samples):
            return [(self.specifications[(int(s.metadata["round_index"]), int(s.metadata["game_id"]))], s.index)
                    for s in samples]
        task = self.pool.apply_async(_prepare_batch, ((specs(drawn.samples), specs(drawn.value_samples)),))
        self.pending.append((drawn, self.rng.snapshot(), task))
        self.remaining -= 1

    def next(self):
        drawn, committed_rng, task = self.pending.popleft()
        batch, value_batch, labels, worker_seconds, pid = task.get()
        if self.remaining:
            self._submit()
        return drawn, committed_rng, batch, value_batch, labels, worker_seconds, pid

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
        self.pending.clear()
        self.cache._entries.clear()
        self.cache._bytes = 0
        self.specifications.clear()
        self.arena.close()
        os.close(self.descriptor)
