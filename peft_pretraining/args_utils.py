import os
import sys
from datetime import datetime

from loguru import logger


class Tee:
    def __init__(self, name, mode="w"):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


def check_args_torchrun_main(args):

    args.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.save_dir is None:
        # use checkpoints / model name, date and time as save directory
        args.save_dir = f"checkpoints/{args.optimizer.lower()}-c4/{args.model_config.split('/')[-1].rstrip('.json')}-{args.desc}"

    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f"save_dir not specified, using {args.save_dir}\n")

    args.log_path = f"{args.save_dir}/log_{args.timestamp}.txt"
    sys.stdout = Tee(args.log_path)
    logger.info(f"Logging to {args.log_path}\n")

    if args.tags is not None:
        args.tags = args.tags.split(",")

    if args.total_batch_size is None:
        args.gradient_accumulation = args.gradient_accumulation or 1
        args.total_batch_size = args.batch_size * args.gradient_accumulation

    assert (
        args.total_batch_size % args.batch_size == 0
    ), "total_batch_size must be divisible by batch_size"

    if args.max_train_tokens is not None:
        args.num_training_steps = args.max_train_tokens // args.total_batch_size
        logger.info(f"Training for {args.num_training_steps} update steps\n")

    if args.continue_from is not None:
        assert os.path.exists(
            args.continue_from
        ), f"--continue_from={args.continue_from} does not exist"

    if args.dtype in ["fp16", "float16"]:
        raise NotImplementedError(
            "fp16 is not supported in torchrun_main.py. Use deepspeed_main.py instead (but it seems to have bugs)"
        )

    return args
