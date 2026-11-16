# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SharedExpertRank must define the attributes every swap mode reads.

Copy-swap mode (VLLM_V41_SWAP_FREE=0 without VLLM_V41_SWAP_INPLACE) never runs
_init_inplace(), yet wait_jobs() runs on every prefill and read
self.async_evict, so DeepSeek-V4.1 with a cold expert set and SWAP_FREE=0 (the
mode Model Runner V2 requires) died with AttributeError on its first prefill.
Constructing a SharedExpertRank needs the native extension and GPUs, so this
checks the source: the attributes are assigned in __init__ outside the
in-place branch, before any mode-specific setup.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_test

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "vllm/model_executor/layers/quantization/gguf_shared_runtime.py"
)
EVERY_MODE = {"async_evict", "job_lane"}


def _rank_class() -> ast.ClassDef:
    tree = ast.parse(SOURCE.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SharedExpertRank"
    )


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _self_reads(func: ast.FunctionDef) -> set[str]:
    return {
        node.attr
        for node in ast.walk(func)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and isinstance(node.ctx, ast.Load)
    }


def _unconditional_self_assigns(body: list[ast.stmt]) -> set[str]:
    """`self.x = ...` statements reachable without entering an if/for/while."""
    names: set[str] = set()
    for stmt in body:
        if isinstance(stmt, ast.Try):
            names |= _unconditional_self_assigns(stmt.body)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    names.add(target.attr)
    return names


def test_wait_jobs_reads_attributes_set_in_every_swap_mode():
    cls = _rank_class()
    assert "async_evict" in _self_reads(_method(cls, "wait_jobs"))
    assigned = _unconditional_self_assigns(_method(cls, "__init__").body)
    missing = EVERY_MODE - assigned
    assert not missing, (
        f"SharedExpertRank.__init__ must set {sorted(missing)} outside the "
        "in-place branch: copy-swap mode never runs _init_inplace()"
    )
