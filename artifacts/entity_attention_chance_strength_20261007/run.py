"""Fixed evaluation CLI with the same FP32/CPU-thread controls as the target diagnostic."""
import sys

import torch

from splendor.evaluation import main


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    raise SystemExit(main(sys.argv[1:]))
