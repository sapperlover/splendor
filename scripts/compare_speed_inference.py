"""Fixed real archive inputs: CPU, GPU wide eager and GPU bucketed outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference, InferenceConfig
from splendor.selfplay_data import STATE_FIELDS
from splendor.features import FEATURE_SCHEMA_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.scratch.mkdir(parents=True, exist_ok=False)
    files = sorted((args.input / 'games').glob('*.npz'))
    widest = (0, None, None)
    for path in files:
        with np.load(path) as archive:
            widths = np.diff(archive['candidate_offsets'])
            index = int(widths.argmax())
            if int(widths[index]) > widest[0]:
                widest = (int(widths[index]), path, index)
    assert 0 < widest[0] <= 256
    references = [(files[0], 0), (widest[1], widest[2])]
    arrays = {name: [] for name in STATE_FIELDS}
    candidates, widths = [], []
    for path, index in references:
        with np.load(path) as archive:
            for name in STATE_FIELDS:
                arrays[name].append(archive[name][index])
            offsets = archive['candidate_offsets']
            feature = archive['candidate_features'][offsets[index]:offsets[index + 1]]
            candidates.append(feature)
            widths.append(len(feature))
    arrays = {name: np.stack(values) for name, values in arrays.items()}

    def packed(width):
        features = np.zeros((len(widths), width, 82), dtype=np.float32)
        mask = np.zeros((len(widths), width), dtype=np.bool_)
        for i, feature in enumerate(candidates):
            features[i, :len(feature)] = feature
            mask[i, :len(feature)] = True
        return dict(arrays, feature_schema_version=FEATURE_SCHEMA_VERSION,
                    candidate_features=features, candidate_mask=mask)

    bucket = next((w for w in (16, 32, 48, 64) if max(widths) <= w), 256)
    inputs = packed(bucket)
    np.savez(args.scratch / 'inputs.npz', **inputs)
    models = ['runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt',
              'runs/az_loop_seed1/rounds/round-004/checkpoints/final.pt',
              'runs/az_loop_seed1/rounds/round-003/checkpoints/final.pt']
    result = {'scope': 'two fixed real root inputs; numerical verification, not throughput ranking',
              'references': [{'file': str(p), 'decision_index': i} for p, i in references],
              'actual_widths': widths, 'bucket': bucket,
              'input_sha256': hashlib.sha256((args.scratch / 'inputs.npz').read_bytes()).hexdigest(), 'models': {}}
    for path in models:
        cpu = FrozenSearchInference(load_frozen_actor_critic_checkpoint(Path(path), device='cpu'), device='cpu')
        expected_logits, expected_values = cpu.policy_value_native(inputs)
        model = load_frozen_actor_critic_checkpoint(Path(path), device='cuda')
        wide = FrozenSearchInference(model, device='cuda', config=InferenceConfig(
            mode='compiled', actor_projection=True, candidate_buckets=(16, 32, 48), compile_failure_policy='fail_fast'))
        optimized = FrozenSearchInference(model, device='cuda', config=InferenceConfig(
            mode='compiled', actor_projection=True, candidate_buckets=(16, 32, 48, 64), compile_failure_policy='fail_fast'))
        checks = {}
        for name, evaluator, data in [('wide_eager', wide, packed(256)), ('bucketed', optimized, inputs)]:
            logits, values = evaluator.policy_value_native(data)
            delta = max(float(np.max(np.abs(logits[i, :w] - expected_logits[i, :w]))) for i, w in enumerate(widths))
            value_delta = float(np.max(np.abs(values - expected_values)))
            assert delta <= 1e-5 and value_delta <= 1e-5, (path, name, delta, value_delta)
            checks[name] = {'max_absolute_logit_difference': delta, 'max_absolute_value_difference': value_delta,
                'argmax_differences': sum(int(logits[i, :w].argmax() != expected_logits[i, :w].argmax())
                                         for i, w in enumerate(widths)), 'inference': evaluator.stats.to_dict()}
        result['models'][hashlib.sha256(Path(path).read_bytes()).hexdigest()] = checks
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'actual_widths': widths, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
