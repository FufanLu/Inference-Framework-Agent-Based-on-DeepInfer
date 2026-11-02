"""The vLLM backend's pure parts, testable without vLLM or a GPU."""

from dataclasses import dataclass

from leviathan_worker.backends.vllm_backend import (
    KvEventTranslator,
    SpecTracker,
    structured_outputs_kwargs,
)
from leviathan_worker.engine import (
    AllCleared,
    BlockRemoved,
    BlockStored,
    ConstraintKind,
    GenerateRequest,
    Medium,
    OutputConstraint,
)
from leviathan_worker.hashing import block_hashes


# Stand-ins with the field names of vllm.distributed.kv_events (v0.31).
@dataclass
class BlockStored_:  # noqa: N801
    block_hashes: list
    parent_block_hash: object
    token_ids: list
    block_size: int
    medium: str = "GPU"
    group_idx: int | None = None
    kv_cache_spec_kind: str | None = None
    kv_cache_spec_sliding_window: int | None = None


@dataclass
class BlockRemoved_:  # noqa: N801
    block_hashes: list
    medium: str = "GPU"
    group_idx: int | None = None


class AllBlocksCleared:
    pass


# The translator dispatches on the vLLM class name.
BlockStored_.__name__ = "BlockStored"
BlockRemoved_.__name__ = "BlockRemoved"

TOKENS = list(range(1000, 1032))  # 4 blocks of 8


def test_stored_chain_is_rehashed_in_leviathan_convention():
    t = KvEventTranslator()
    first = t.translate([BlockStored_([b"v0", b"v1"], None, TOKENS[:16], 8)])
    second = t.translate([BlockStored_([b"v2", b"v3"], b"v1", TOKENS[16:], 8)])
    expected = block_hashes(TOKENS, 8)
    assert first == [BlockStored(tuple(expected[:2]), None, Medium.GPU)]
    assert second == [BlockStored(tuple(expected[2:]), expected[1], Medium.GPU)]


def test_removal_maps_back_and_unknown_parents_are_orphans():
    t = KvEventTranslator()
    t.translate([BlockStored_([7, 8], None, TOKENS[:16], 8)])
    expected = block_hashes(TOKENS[:16], 8)
    assert t.translate([BlockRemoved_([8, 99], medium="CPU")]) == [BlockRemoved((expected[1],), Medium.CPU)]
    assert t.translate([BlockStored_([5], 12345, TOKENS[:8], 8)]) == []
    assert t.orphans == 1


def test_clear_resets_the_mapping():
    t = KvEventTranslator()
    t.translate([BlockStored_([1], None, TOKENS[:8], 8)])
    assert t.translate([AllBlocksCleared()]) == [AllCleared()]
    assert t.translate([BlockRemoved_([1])]) == []


def test_sliding_window_groups_are_ignored():
    t = KvEventTranslator()
    swa = BlockStored_([1], None, TOKENS[:8], 8, group_idx=1, kv_cache_spec_kind="sliding_window", kv_cache_spec_sliding_window=128)
    full = BlockStored_([2], None, TOKENS[:8], 8, group_idx=0, kv_cache_spec_kind="mla_attention")
    assert t.translate([swa]) == []
    assert len(t.translate([full])) == 1
    assert t.group == 0


def test_spec_tracker_reports_deltas():
    @dataclass
    class Metrics:
        num_draft_tokens: int
        histogram: list

    tracker = SpecTracker()
    assert tracker.step(Metrics(5, [0, 0, 1])) == (5, 2)
    assert tracker.step(Metrics(10, [0, 1, 1])) == (5, 1)
    assert tracker.step(None) == (0, 0)


def test_constraints_map_to_structured_outputs():
    def request(kind, schema=""):
        return GenerateRequest(request_id="r", prompt_token_ids=(1,), constraint=OutputConstraint(kind, schema))

    assert structured_outputs_kwargs(request(ConstraintKind.JSON, '{"type": "object"}')) == {"json": {"type": "object"}}
    assert structured_outputs_kwargs(request(ConstraintKind.JSON)) == {"json_object": True}
    assert structured_outputs_kwargs(request(ConstraintKind.REGEX, "[a-z]+")) == {"regex": "[a-z]+"}
    assert structured_outputs_kwargs(request(ConstraintKind.TOOL_CALL, "[]")) is None
