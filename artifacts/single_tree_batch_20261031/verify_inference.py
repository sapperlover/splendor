"""Compare the deployed model's real native FP32 inputs, including payment/return."""
import json
from pathlib import Path

import numpy as np
import torch

from splendor.compact_trace import decode_request
from splendor.kirameki_live import DEFAULT_CHECKPOINT, request_from_snapshot
from splendor.native_policy import NativeBatchRunner, NativeGreedyPolicy
from splendor.recommend import LocalAdvisor
from splendor.search_inference import FrozenSearchInference, InferenceConfig


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    advisor=LocalAdvisor(DEFAULT_CHECKPOINT,device='cuda')
    compiled=FrozenSearchInference(advisor.model,device='cuda',config=InferenceConfig(
        mode='compiled',compile_mode='native-cudagraph',candidate_buckets=(16,32,48,64),
        batch_buckets=(1,64),compile_failure_policy='fail_fast'))
    policy=NativeGreedyPolicy(advisor.model,evaluator=compiled)
    runner=NativeBatchRunner(compiled,advisor.catalog,capacity=64,buckets=(16,32,48,64))
    runner.warmup()
    base=Path('runtime/kirameki_budget_matrix_20261030');requests=[]
    cases=['g03_r002_main','g04_r013_main','g05_r024_main','g03_r007_followup','g03_r031_followup']
    for case in cases:
        spec=json.loads((base/'results'/f'{case}.json').read_text())['spec']
        directory=base/spec['input_directory']
        requests.append(request_from_snapshot(json.loads((directory/'snapshot.json').read_text()))
            if spec['kind']=='main' else
            decode_request(json.loads((directory/'recommendation.json').read_text())['followup']['request']))
    batch,transfer=runner._buffer(0,False,64)
    for i in range(64):assert batch.fill(policy.start_job(requests[i%len(requests)]).core)==1
    with torch.inference_mode():
        expected_logits,expected_values=advisor.evaluator._forward(
            advisor.evaluator._tensor_batch_native(batch.arrays(),state_only=False))
        expected_logits=expected_logits.cpu().numpy();expected_values=expected_values.cpu().numpy()
    handle=transfer.begin_policy_value(batch);handle._event.synchronize()
    actual=transfer.out_numpy[:64].copy();max_logit=0.
    for row in range(64):
        count=len(requests[row%len(requests)].legal_actions)
        np.testing.assert_allclose(actual[row,:count],expected_logits[row,:count],atol=1e-5,rtol=1e-4)
        max_logit=max(max_logit,float(np.max(np.abs(actual[row,:count]-expected_logits[row,:count]))))
    np.testing.assert_allclose(actual[:,64],expected_values,atol=1e-5,rtol=1e-4)
    handle.finish()
    result=dict(passed=True,checkpoint_sha256=advisor.checkpoint_sha256,cases=cases,rows=64,
        max_abs_logit_error=max_logit,max_abs_value_error=float(np.max(np.abs(actual[:,64]-expected_values))),
        inference=compiled.stats.to_dict())
    Path('runtime/single_tree_batch_20261031/inference_verification.json').write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='inference'},ensure_ascii=False))


if __name__=='__main__':main()
