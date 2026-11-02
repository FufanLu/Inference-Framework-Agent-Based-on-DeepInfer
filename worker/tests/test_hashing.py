from leviathan_worker.hashing import ROOT_SEED, block_hashes

# Shared with crates/lev-router/src/hashing.rs (CROSS_LANGUAGE_VECTOR).
CROSS_LANGUAGE_VECTOR = [0x05CA_F1E6_F5C0_6121, 0x075F_F973_CE99_7C6B]


def test_matches_cross_language_vector():
    assert block_hashes(list(range(10)), 4) == CROSS_LANGUAGE_VECTOR


def test_only_full_blocks():
    assert block_hashes([1, 2, 3], 4) == []
    assert len(block_hashes(list(range(10)), 4)) == 2


def test_chain_continuation_equals_full_computation():
    tokens = list(range(100, 132))
    full = block_hashes(tokens, 8)
    head = block_hashes(tokens[:16], 8)
    tail = block_hashes(tokens[16:], 8, parent=head[-1])
    assert head + tail == full


def test_root_seed_spells_leviathan():
    assert ROOT_SEED.to_bytes(8, "big") == b"LEVIATHA"
