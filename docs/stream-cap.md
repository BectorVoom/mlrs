# The CubeCL stream cap

mlrs pins CubeCL's `streaming.max_streams` to **1** before the first compute
client is built (`mlrs_backend::stream_cap`, installed by
`runtime::active_client`). This page says why, and how to change it.

## Why

CubeCL allocates **one GPU stream per OS thread**. `StreamId::current()` reads a
thread-local drawn from a global counter, and `StreamPool` indexes streams as
`stream_id.value % max_streams`, where `max_streams` defaults to **128**.
Critically, every `Stream` owns an *independent* `MemoryManagement<GpuStorage>`:
a device page belongs to exactly one stream's arena, and streams are never torn
down.

That default is wrong for mlrs in both directions.

**It cannot help.** Every device call in the Python extension runs while holding
the process-global `Mutex<BufferPool>` (`crates/mlrs-py/src/lib.rs`), so mlrs
never has two kernels in flight from two threads. Extra streams buy exactly zero
overlap.

**It costs a great deal.** Any thread fan-out over mlrs estimators — joblib's
`threading` backend under `StackingRegressor(n_jobs=...)`, a threaded
`GridSearchCV`, a user's own `ThreadPoolExecutor` — mints a fresh thread, and
therefore a fresh stream and a fresh arena, per `Parallel` call (joblib's
threading backend does not reuse threads across calls). Buffers mlrs's own pool
would have reused are re-allocated per stream, and every launch then pays
cross-stream flush/wait alignment.

On rocm gfx1151 — where the VRAM carve-out is 512 MB and already ~96% used at
idle — that reliably exhausted the device heap and killed the process.

## The failure it fixes

First panic, on the shared `DSD-0-0` server thread:

```
cubecl-hip/src/compute/server.rs:84   HipServer::initialize_memory
    command.reserve(size).unwrap()  ->  "Unknown error happened during execution"
```

reached from `do_create` — an upload that could not reserve device memory.
Everything after it (`"Memory page N doesn't exist"` at `stream.rs:127`,
`create_with_data` at `server.rs:388`, `CallError` at
`cubecl-runtime/src/client.rs:105`) is the cascade once that thread is already
unwinding. **Read such a log bottom-up**: the page-lookup failures are a
consequence, and chasing them first sends you after handle provenance, which is
correct in CubeCL and a dead end.

Measured on the reproducer — a six-fit stacking script at n=60000, d=48:

| `max_streams` | outcome |
|---|---|
| 128 (cubecl default) | **aborts** deterministically, 3/3 runs |
| 16 / 8 / 4 / 2 / 1 | survives; every result bit-identical to serial |

Capping is also *faster*, because the cross-stream alignment disappears: cv=5 at
`n_jobs=4` went ~3.2 s (before dying) to ~0.24 s.

mlrs's own `BufferPool` was never the culprit and is working well — 300
single-threaded 11.5 MB round-trips cost exactly **1 allocation and 599 reuses**.
But its free-list is keyed by byte size alone, so a handle it hands to a new
thread still belongs to the *old* stream; the new stream allocates its own pages
regardless.

The full investigation, including the eight probes that ruled out handle
provenance, is in `crates/mlrs-backend/tests/thread_stream_probe_test.rs`.

## Precedence

`stream_cap::install()` runs once, before the first client, and is deliberately
conservative about overriding a deliberate choice:

1. `MLRS_MAX_STREAMS=<n>` in the environment wins outright. Set
   `MLRS_MAX_STREAMS=128` to restore CubeCL's default (and reproduce the old
   behaviour). A value that is not an integer in `1..=255` is ignored with a
   warning — `0` in particular, since `stream_id % 0` would panic inside CubeCL.
2. Otherwise, a `cubecl.toml` that sets `streaming.max_streams` to anything
   other than CubeCL's own default is the user's decision and is respected.
   Every other `CUBECL_*` env override and `cubecl.toml` key is loaded normally.
3. Otherwise the cap applies.

If some other code reads or sets the CubeCL config before the first
`active_client()` call, the cap cannot be applied — CubeCL's config is
write-once. `stream_cap` leaves the existing value alone and logs a warning when
it is above the cap.

## Scope

The cap is backend-independent: it is host configuration in `cubecl-runtime`,
and the same per-thread-stream design applies to cuda and wgpu. It is applied on
every backend rather than gated to rocm, because the reasoning — mlrs serializes
device work behind one mutex — does not depend on the backend, and because a
latent OOM that only appears on one vendor's hardware is exactly the kind of bug
this cost a day to find.

Tests: `crates/mlrs-backend/tests/stream_cap_test.rs`.
