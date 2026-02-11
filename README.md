# Vexor

High-performance in-memory vector similarity search engine written in pure Go. Zero external dependencies.

On an Apple M4 Pro (14 cores), Vexor achieves **3,900+ QPS** on 100k 128-dimensional vectors with **~255 us P50 latency** — a **5.2x throughput improvement** over a single-core scalar baseline.

## Features

- **ARM NEON SIMD** — Hand-written assembly with 4-accumulator unrolling for distance computations (3.0-3.8x speedup over scalar Go)
- **Sharded parallel search** — 16 FNV-hashed shards with per-shard `RWMutex`, goroutine-parallel k-NN search
- **SoA memory layout** — Contiguous `[]float32` storage per shard for cache-friendly sequential scans (4.9x over AoS)
- **Distance metrics** — Euclidean, dot product, cosine similarity
- **O(1) deletion** — Swap-with-last backed by an ID index map
- **Upsert** — Insert with existing ID updates in-place

## Quick Start

```bash
# Build
go build ./...

# Run demo (10k vectors, sample query)
go run ./cmd/main.go

# Run tests
go test ./...

# Run benchmarks
go test -bench=. -benchmem ./bench/
go test -v -run TestQPSAndLatency ./bench/
go test -v -run TestFullReport ./bench/
```

## Performance

Benchmarked on Apple M4 Pro, 14 cores, Go 1.25.6, 100k vectors, dim=128, k=10.

| Metric | Value |
|---|---|
| QPS | 3,904 |
| P50 Latency | 253.6 us |
| P99 Latency | 300.2 us |

| SIMD Function | Scalar (ns/op) | NEON (ns/op) | Speedup |
|---|---:|---:|---:|
| EuclideanDistanceSquared | 55.8 | 14.6 | 3.82x |
| DotProduct | 36.7 | 12.1 | 3.04x |

Scaling plateaus at ~8 cores due to the 16-shard design and goroutine overhead. See [`doc/performance-report.md`](doc/performance-report.md) for the full analysis.

## Architecture

```
pkg/distance/   Distance functions with NEON assembly (arm64) and scalar fallback
pkg/store/      Sharded vector store with SoA layout and parallel k-NN search
cmd/            Demo entrypoint
bench/          Benchmarks (QPS, latency, SIMD, AoS vs SoA, core scaling)
doc/            Performance report
```
