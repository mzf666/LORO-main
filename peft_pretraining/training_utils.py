import math
from functools import partial

import torch
from torch.optim.lr_scheduler import LambdaLR
import transformers


def get_scheduler(
    optimizer,
    *,
    scheduler_type,
    num_training_steps,
    warmup_steps,
    min_lr_ratio,
    cycle_length=None,
    restart_warmup_steps=None,
    lr_adjust_steps=0,
    lr_jag_after_warmup=False,
    last_epoch=-1,
):
    if lr_adjust_steps != 0 and (
        not scheduler_type in ["cosine_restart", "cosine_restart_zero"]
    ):
        raise ValueError(
            "lr_adjust_steps is only supported for cosine_restart* scheduler"
        )

    if scheduler_type == "const":
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1, total_iters=num_training_steps
        )

    if scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
    if scheduler_type == "cosine":
        return get_cyclical_cosine_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            cycle_length=cycle_length,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
        )
    if scheduler_type == "cosine_restart":
        assert (
            restart_warmup_steps is not None
        ), "restart_warmup_steps must be specified for cosine_restart scheduler"
        return get_cosine_schedule_with_multiple_warmups(
            optimizer,
            num_training_steps=num_training_steps,
            first_warmup_steps=warmup_steps,
            restart_warmup_steps=restart_warmup_steps,
            restart_every=cycle_length,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
            lr_adjust_steps=lr_adjust_steps,
            lr_jag_after_warmup=lr_jag_after_warmup,
        )

    if scheduler_type == "cosine_restart_zero":
        assert (
            restart_warmup_steps is not None
        ), "restart_warmup_steps must be specified for cosine_restart scheduler"
        return get_cosine_schedule_with_multiple_warmups_from_zero(
            optimizer,
            num_training_steps=num_training_steps,
            first_warmup_steps=warmup_steps,
            restart_warmup_steps=restart_warmup_steps,
            restart_every=cycle_length,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
            lr_adjust_steps=lr_adjust_steps,
            lr_jag_after_warmup=lr_jag_after_warmup,
        )

    raise NotImplementedError(f"Scheduler {scheduler_type} is not implemented")


def get_cyclical_cosine_schedule_with_min_lr(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    cycle_length,
    min_lr_ratio=0.1,
    last_epoch=-1,
):
    assert (
        cycle_length is not None or num_training_steps is not None
    ), "You must specify either cycle_length or num_training_steps"

    if cycle_length is None:
        cycle_length = num_training_steps

    if num_training_steps % cycle_length != 0:
        raise ValueError(
            f"num_training_steps ({num_training_steps}) must be divisible by cycle_length ({cycle_length})"
        )

    lr_lambda = partial(
        _get_cyclical_cosine_schedule_with_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        cycle_length=cycle_length,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_multiple_warmups(
    optimizer,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio=0.1,
    lr_adjust_steps=0,
    last_epoch=-1,
    lr_jag_after_warmup=False,
):
    if restart_every is None:
        raise ValueError("restart_every must be specified for cosine_restart scheduler")

    # if num_training_steps % restart_every != 0:
    #     raise ValueError(
    #         f"num_training_steps ({num_training_steps}) must be divisible by restart_every ({restart_every})"
    #     )

    lr_lambda = partial(
        _get_cosine_schedule_with_multiple_warmups_lambda,
        num_training_steps=num_training_steps,
        first_warmup_steps=first_warmup_steps,
        restart_warmup_steps=restart_warmup_steps,
        restart_every=restart_every,
        min_lr_ratio=min_lr_ratio,
        lr_adjust_steps=lr_adjust_steps,
        lr_jag_after_warmup=lr_jag_after_warmup,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_multiple_warmups_from_zero(
    optimizer,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio=0.1,
    lr_adjust_steps=0,
    last_epoch=-1,
    lr_jag_after_warmup=False,
):
    if restart_every is None:
        raise ValueError("restart_every must be specified for cosine_restart scheduler")

    # if num_training_steps % restart_every != 0:
    #     raise ValueError(
    #         f"num_training_steps ({num_training_steps}) must be divisible by restart_every ({restart_every})"
    #     )

    lr_lambda = partial(
        _get_cosine_schedule_with_multiple_warmups_from_zero_lambda,
        num_training_steps=num_training_steps,
        first_warmup_steps=first_warmup_steps,
        restart_warmup_steps=restart_warmup_steps,
        restart_every=restart_every,
        min_lr_ratio=min_lr_ratio,
        lr_adjust_steps=lr_adjust_steps,
        lr_jag_after_warmup=lr_jag_after_warmup,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


@torch.no_grad()
def random_pruning(tensor, prune_ratio):
    """
    Performs random pruning dimensionality reduction.
    Only reduces the inner dimensionality, does not affect the shape of the tensor
    """
    random_pruning_mask = torch.rand_like(tensor) > prune_ratio
    tensor = tensor * random_pruning_mask
    return tensor


@torch.no_grad()
def magnitude_pruning(tensor, prune_ratio):
    """
    Performs magnitude pruning dimensionality reduction.
    Only reduces the inner dimensionality, does not affect the shape of the tensor
    """
    tensor_magnitude = torch.abs(tensor)
    threshold = torch.quantile(
        tensor_magnitude.flatten().to(dtype=torch.float32), prune_ratio
    ).to(dtype=tensor.dtype)

    mask = tensor_magnitude > threshold
    tensor = tensor * mask.to(dtype=tensor.dtype)
    return tensor


def _get_cyclical_cosine_schedule_with_min_lr_lambda(
    current_step, *, num_warmup_steps, cycle_length, min_lr_ratio
):
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"

    # compute where we are in the current cycle
    cycle_step = current_step % cycle_length

    if cycle_step < num_warmup_steps:
        if current_step != cycle_step:
            if cycle_step < 2:
                return 1e-7
        return float(cycle_step) / float(max(1, num_warmup_steps))

    progress = float(cycle_step - num_warmup_steps) / float(
        max(1, cycle_length - num_warmup_steps)
    )
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def _get_cosine_schedule_with_multiple_warmups_lambda(
    current_step,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio,
    lr_adjust_steps,
    lr_jag_after_warmup,
    eps=1e-6,
):
    """
    Args:
        lr_adjust_steps: useful when continuing training from a warmed up checkpoint,
            it allows to sync the resets by reducing the number of steps
            after the first warmup and before the first reset.
            Thus, your ReLoRA resets can be synced with the optimizer resets.
    """
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"
    assert restart_every > 0, "restart_every must be positive"
    assert (
        lr_adjust_steps + first_warmup_steps < num_training_steps
    ), "warmup + lr_adjust_steps is more than full training steps"
    # assert (
    #     lr_adjust_steps + first_warmup_steps < restart_every
    # ), "the first reset will happen before the warmup is done"

    if current_step < first_warmup_steps:

        max_warmup_lr = 0.5 * (
            1.0
            + math.cos(
                math.pi
                * float(-lr_adjust_steps)
                / float(max(1, num_training_steps - first_warmup_steps))
            )
        )

        _current_step = current_step

        restart_step = _current_step % restart_every
        restart_number = _current_step // restart_every

        if (
            restart_step < restart_warmup_steps
            and restart_number != 0
            and not lr_jag_after_warmup
        ):
            # get expected lr multipler at the end of the warmup
            end_of_warmup_progress = float(restart_number * restart_every) / float(
                max(1, first_warmup_steps)
            )

            # warmup_lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * end_of_warmup_progress
            warmup_lr_multiplier = (
                end_of_warmup_progress * (max_warmup_lr - min_lr_ratio) + min_lr_ratio
            )

            return (
                float(restart_step)
                / float(max(1, restart_warmup_steps))
                * (warmup_lr_multiplier - eps)
                + eps
            )

        return (
            float(_current_step)
            / float(max(1, first_warmup_steps))
            * (max_warmup_lr - min_lr_ratio)
            + min_lr_ratio
        )

    else:

        _current_step = current_step + lr_adjust_steps

        restart_step = _current_step % restart_every
        restart_number = _current_step // restart_every

        if restart_step < restart_warmup_steps and restart_number > 0:
            # get expected lr multipler at the end of the warmup
            end_of_warmup_progress = float(
                restart_number * restart_every - first_warmup_steps
            ) / float(max(1, num_training_steps - first_warmup_steps))

            _cosine_decay = 0.5 * (1.0 + math.cos(math.pi * end_of_warmup_progress))
            warmup_lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * _cosine_decay

            return (
                float(restart_step)
                / float(max(1, restart_warmup_steps))
                * (warmup_lr_multiplier - eps)
                + eps
            )

        progress = float(_current_step - first_warmup_steps) / float(
            max(1, num_training_steps - first_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def _get_cosine_schedule_with_multiple_warmups_from_zero_lambda(
    current_step,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio,
    lr_adjust_steps,
    lr_jag_after_warmup,
    eps=1e-6,
):
    """
    Args:
        lr_adjust_steps: useful when continuing training from a warmed up checkpoint,
            it allows to sync the resets by reducing the number of steps
            after the first warmup and before the first reset.
            Thus, your ReLoRA resets can be synced with the optimizer resets.
    """
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"
    assert restart_every > 0, "restart_every must be positive"
    assert (
        lr_adjust_steps + first_warmup_steps < num_training_steps
    ), "warmup + lr_adjust_steps is more than full training steps"
    # assert (
    #     lr_adjust_steps + first_warmup_steps < restart_every
    # ), "the first reset will happen before the warmup is done"

    if current_step < first_warmup_steps:

        max_warmup_lr = 0.5 * (
            1.0
            + math.cos(
                math.pi
                * float(-lr_adjust_steps)
                / float(max(1, num_training_steps - first_warmup_steps))
            )
        )

        _current_step = current_step

        restart_step = _current_step % restart_every
        restart_number = _current_step // restart_every

        if (
            restart_step < restart_warmup_steps
            and restart_number != 0
            and not lr_jag_after_warmup
        ):
            # get expected lr multipler at the end of the warmup
            end_of_warmup_progress = float((restart_number) * restart_every) / float(
                max(1, first_warmup_steps)
            )

            # warmup_lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * end_of_warmup_progress
            warmup_lr_multiplier = end_of_warmup_progress * (max_warmup_lr)

            eps = warmup_lr_multiplier * 0.1
            return (
                float(restart_step)
                / float(max(1, restart_warmup_steps))
                * (warmup_lr_multiplier - eps)
                + eps
            )

        return (
            float(_current_step) / float(max(1, first_warmup_steps)) * (max_warmup_lr)
        )

    else:

        _current_step = current_step + lr_adjust_steps

        restart_step = _current_step % restart_every
        restart_number = _current_step // restart_every

        if (restart_step < restart_warmup_steps) and (restart_number != 0):
            # get expected lr multipler at the end of the warmup
            end_of_warmup_progress = float(
                restart_number * restart_every - first_warmup_steps
            ) / float(max(1, num_training_steps - first_warmup_steps))

            _cosine_decay = 0.5 * (1.0 + math.cos(math.pi * end_of_warmup_progress))
            warmup_lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * _cosine_decay

            eps = warmup_lr_multiplier * 0.1

            return (
                float(restart_step)
                / float(max(1, restart_warmup_steps))
                * (warmup_lr_multiplier - eps)
                + eps
            )

        progress = float(_current_step - first_warmup_steps) / float(
            max(1, num_training_steps - first_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def collate_fn(batch_list):
    batch = {
        "input_ids": torch.stack(
            [torch.Tensor(example["input_ids"]).long() for example in batch_list]
        ),
        "attention_mask": torch.stack(
            [torch.Tensor(example["attention_mask"]).long() for example in batch_list]
        ),
    }
    return batch


def batch_fn(dataset, batch_size):
    batch = []
    for example in dataset:
        batch.append(example)
        if len(batch) == batch_size:
            batch = collate_fn(batch)
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def max_train_tokens_to_number(max_train_tokens):
    if max_train_tokens.endswith("M"):
        return int(max_train_tokens.rstrip("M")) * 1_000_000
    elif max_train_tokens.endswith("B"):
        return int(max_train_tokens.rstrip("B")) * 1_000_000_000
    else:
        return int(max_train_tokens)
