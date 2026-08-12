# safetensors 0.8.0 — mlrs fork

Upstream: <https://github.com/huggingface/safetensors> · Apache-2.0 (see `LICENSE`)

This is an **unmodified copy of the published `safetensors` 0.8.0 crate plus one
patch**, wired into the workspace through `[patch.crates-io]` in the root
`Cargo.toml`. It exists for a single reason:

> **Make a saved model a deterministic function of its contents.**

This file (`FORK.md`) documents the fork; `README.md` is upstream's own, kept
verbatim because `lib.rs` includes it as crate documentation.

`patches/0001-btreemap-metadata.patch` is the whole diff, regenerated against the
pristine 0.8.0 release. Nothing else in this directory differs from upstream, so
the fork can be re-derived, audited, or dropped mechanically.

## The defect

`safetensors` writes `u64 header_len | JSON header | payload`. `Metadata`'s
`Serialize` impl emits tensor entries through an index map, so those come out in
a stable order — but it hands the free-form `__metadata__` map straight to
`serde_json`:

```rust
let length = self.metadata.as_ref().map_or(0, HashMap::len);
let mut map = serializer.serialize_map(Some(self.tensors.len() + length))?;
if let Some(metadata) = &self.metadata {
    map.serialize_entry("__metadata__", metadata)?;   // <-- std HashMap
}
```

`std::collections::HashMap` iterates in an order derived from a randomly seeded
hasher, so **saving the same model twice produces different bytes**. Only the
`__metadata__` keys shuffle; every tensor payload is already byte-stable.

Observed on mlrs Naive Bayes model files — two saves of one `GaussianNB`:

```
save A: {"__metadata__":{"param:var_smoothing":"1e-9","version":"1","format":"mlrs-nb", ...
save B: {"__metadata__":{"format":"mlrs-nb","estimator":"gaussian_nb","version":"1", ...
```

Identical payloads, identical header length, different bytes.

## Why a patch and not a workaround

The map is taken **by value in the public signatures**:

```rust
pub fn serialize    (data: I, data_info: Option<HashMap<String, String>>)          -> ...
pub fn serialize_to_file(data: I, data_info: Option<HashMap<String, String>>, ...) -> ...
```

A caller therefore cannot substitute an ordered map, sort the keys, or hook the
serializer. The alternatives were to pack every scalar into one JSON-encoded
`__metadata__` entry (deterministic, but it destroys the plain key/value view
that makes these files readable from `safetensors.numpy.load_file`), or to
hand-roll the container writer (which would mean owning the dtype-descending
tensor ordering that mlrs's zero-copy aligned reads depend on). Both are worse
than a nine-line retype.

## The patch

Every `__metadata__` map becomes a `BTreeMap`, which iterates in sorted key
order:

| site | change |
|---|---|
| `lib::{no_stds, stds}` | export `BTreeMap` alongside `HashMap` |
| `prepare`, `serialize`, `serialize_to_file` | `data_info: Option<BTreeMap<String, String>>` |
| `Metadata::metadata`, `HashMetadata::metadata` | field type |
| `Metadata::new`, `Metadata::metadata()` | parameter / return type |
| `Serialize for Metadata` | `map_or(0, BTreeMap::len)` |
| two in-crate tests | construct a `BTreeMap` |

`HashMetadata::tensors` and `Metadata::index_map` stay `HashMap` — neither
affects output order.

`no_std` is preserved: `alloc::collections::BTreeMap` needs no `hashbrown`.

**This is a breaking API change** for anyone passing a `HashMap` to `serialize`
/ `serialize_to_file`, which is why upstream would need a version bump to take
it. A non-breaking variant exists — keep the public `HashMap` and collect into a
transient `BTreeMap` inside `Serialize for Metadata` — at the cost of one small
allocation per serialize. If you upstream this, offer that variant first; it is
likelier to land.

## Verification

Upstream's own suite passes unchanged against the patch:

```
cargo test --manifest-path third_party/safetensors/Cargo.toml
# 36 passed (lib) + 4 passed (doc)
```

And on the mlrs side, `crates/mlrs-algos/tests/nb_persist_test.rs` gates it from
two directions:

- `saving_twice_produces_an_identical_model` — raw file bytes must match;
- `metadata_keys_are_written_in_sorted_order` — the header literally spells the
  keys out sorted, so the test cannot pass by landing on the same random order
  twice.

Byte-determinism was also confirmed across three separate processes (identical
md5 per model), which is the case a single-process test cannot cover, since
`RandomState`'s seed is per-process.

## Removing this fork

When the change lands in a published `safetensors`:

1. delete the `[patch.crates-io]` block and the `exclude` entry in the root
   `Cargo.toml`;
2. bump the `safetensors` pin in `[workspace.dependencies]`;
3. delete this directory;
4. run the two tests above — they are the regression gate, and they fail loudly
   if the release does not actually carry the fix.

## Caveat for downstream consumers

`[patch]` is **not** inherited through a published crate. Anything depending on
mlrs from crates.io resolves stock `safetensors` and loses determinism. The mlrs
wheels are built from this workspace, so they get the fix.
