# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from slam_llm.utils.dataset_utils import *  # noqa: F401,F403
from slam_llm.utils.fsdp_utils import fsdp_auto_wrap_policy  # noqa: F401
from slam_llm.utils.train_utils import *  # noqa: F401,F403

try:
    from slam_llm.utils.memory_utils import MemoryTrace  # type: ignore
except Exception:
    class MemoryTrace:  # type: ignore

        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def log(self, *args, **kwargs):
            pass