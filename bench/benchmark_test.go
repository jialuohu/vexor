package bench

import (
	"container/heap"
	"fmt"
	"math/rand"
	"runtime"
	"sort"
	"testing"
	"time"

	"vexor/pkg/distance"
	"vexor/pkg/store"
)

const (
	numVectors = 100_000
	numQueries = 1_000
	dimension  = 128
	k          = 10
)

// generateRandomVector creates a random float32 vector of given dimension.
func generateRandomVector(dim int, rng *rand.Rand) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}
	return v
}

// BenchmarkSearch benchmarks the k-NN search.
func BenchmarkSearch(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	s := store.NewVectorStore(dimension)

	for i := 0; i < numVectors; i++ {
		s.Insert(store.Vector{
			ID:   fmt.Sprintf("vec-%d", i),
			Data: generateRandomVector(dimension, rng),
		})
	}

	queries := make([][]float32, numQueries)
	for i := range queries {
		queries[i] = generateRandomVector(dimension, rng)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		query := queries[i%numQueries]
		s.Search(query, k)
	}
}

// BenchmarkSearchCosine benchmarks cosine similarity search.
func BenchmarkSearchCosine(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	s := store.NewVectorStore(dimension)

	for i := 0; i < numVectors; i++ {
		s.Insert(store.Vector{
			ID:   fmt.Sprintf("vec-%d", i),
			Data: generateRandomVector(dimension, rng),
		})
	}

	queries := make([][]float32, numQueries)
	for i := range queries {
		queries[i] = generateRandomVector(dimension, rng)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		query := queries[i%numQueries]
		s.SearchCosine(query, k)
	}
}

// TestQPSAndLatency measures QPS and latency metrics.
func TestQPSAndLatency(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping QPS test in short mode")
	}

	rng := rand.New(rand.NewSource(42))
	s := store.NewVectorStore(dimension)

	t.Logf("Inserting %d vectors of dimension %d...", numVectors, dimension)
	insertStart := time.Now()
	for i := 0; i < numVectors; i++ {
		s.Insert(store.Vector{
			ID:   fmt.Sprintf("vec-%d", i),
			Data: generateRandomVector(dimension, rng),
		})
	}
	insertDuration := time.Since(insertStart)
	t.Logf("Insert completed in %v (%.0f vectors/sec)", insertDuration, float64(numVectors)/insertDuration.Seconds())

	queries := make([][]float32, numQueries)
	for i := range queries {
		queries[i] = generateRandomVector(dimension, rng)
	}

	t.Logf("Running %d queries (k=%d)...", numQueries, k)
	latencies := make([]time.Duration, numQueries)
	queryStart := time.Now()

	for i, query := range queries {
		start := time.Now()
		_, err := s.Search(query, k)
		latencies[i] = time.Since(start)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
	}

	totalDuration := time.Since(queryStart)
	qps := float64(numQueries) / totalDuration.Seconds()

	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	p50, p99, p999 := percentiles(latencies)

	t.Logf("\n=== Performance Report ===")
	t.Logf("Vectors: %d, Dimension: %d, k: %d", numVectors, dimension, k)
	t.Logf("GOMAXPROCS: %d", runtime.GOMAXPROCS(0))
	t.Logf("-----------------------------------")
	t.Logf("QPS:           %.2f", qps)
	t.Logf("P50 Latency:   %v", p50)
	t.Logf("P99 Latency:   %v", p99)
	t.Logf("P99.9 Latency: %v", p999)
	t.Logf("===================================")
}

// BenchmarkInsert benchmarks vector insertion.
func BenchmarkInsert(b *testing.B) {
	rng := rand.New(rand.NewSource(42))

	vectors := make([]store.Vector, b.N)
	for i := range vectors {
		vectors[i] = store.Vector{
			ID:   fmt.Sprintf("vec-%d", i),
			Data: generateRandomVector(dimension, rng),
		}
	}

	s := store.NewVectorStore(dimension)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		s.Insert(vectors[i])
	}
}

// TestFullReport produces a comprehensive performance comparison across all optimization levels.
func TestFullReport(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping full report in short mode")
	}

	rng := rand.New(rand.NewSource(42))

	// --- Section 1: Distance function micro-benchmarks ---
	t.Log("\n============================================================")
	t.Log("         VEXOR FULL PERFORMANCE REPORT")
	t.Log("============================================================")
	t.Logf("CPU: %d cores available", runtime.NumCPU())
	t.Logf("GOMAXPROCS: %d", runtime.GOMAXPROCS(0))
	t.Logf("Vectors: %d, Dimension: %d, k: %d", numVectors, dimension, k)

	// Generate test vectors for micro-benchmarks
	a := generateRandomVector(dimension, rng)
	b := generateRandomVector(dimension, rng)

	const distIters = 1_000_000

	// Scalar EuclideanDistanceSquared
	start := time.Now()
	for i := 0; i < distIters; i++ {
		distance.EuclideanDistanceSquaredScalar(a, b)
	}
	scalarEucTime := time.Since(start)

	// NEON EuclideanDistanceSquared
	start = time.Now()
	for i := 0; i < distIters; i++ {
		distance.EuclideanDistanceSquared(a, b)
	}
	neonEucTime := time.Since(start)

	// Scalar DotProduct
	start = time.Now()
	for i := 0; i < distIters; i++ {
		distance.DotProductScalar(a, b)
	}
	scalarDotTime := time.Since(start)

	// NEON DotProduct
	start = time.Now()
	for i := 0; i < distIters; i++ {
		distance.DotProduct(a, b)
	}
	neonDotTime := time.Since(start)

	t.Log("\n--- Section 1: SIMD Impact (Scalar vs NEON) ---")
	t.Logf("EuclideanDistanceSquared (128-dim, %d iterations):", distIters)
	t.Logf("  Scalar: %v  (%.1f ns/op)", scalarEucTime, float64(scalarEucTime.Nanoseconds())/float64(distIters))
	t.Logf("  NEON:   %v  (%.1f ns/op)", neonEucTime, float64(neonEucTime.Nanoseconds())/float64(distIters))
	t.Logf("  Speedup: %.2fx", float64(scalarEucTime)/float64(neonEucTime))
	t.Logf("DotProduct (128-dim, %d iterations):", distIters)
	t.Logf("  Scalar: %v  (%.1f ns/op)", scalarDotTime, float64(scalarDotTime.Nanoseconds())/float64(distIters))
	t.Logf("  NEON:   %v  (%.1f ns/op)", neonDotTime, float64(neonDotTime.Nanoseconds())/float64(distIters))
	t.Logf("  Speedup: %.2fx", float64(scalarDotTime)/float64(neonDotTime))

	// --- Section 2: Full search QPS/latency ---
	s := store.NewVectorStore(dimension)

	t.Logf("\nInserting %d vectors...", numVectors)
	for i := 0; i < numVectors; i++ {
		s.Insert(store.Vector{
			ID:   fmt.Sprintf("vec-%d", i),
			Data: generateRandomVector(dimension, rng),
		})
	}

	queries := make([][]float32, numQueries)
	for i := range queries {
		queries[i] = generateRandomVector(dimension, rng)
	}

	// Warmup
	for i := 0; i < 10; i++ {
		s.Search(queries[i], k)
	}

	latencies := make([]time.Duration, numQueries)
	queryStart := time.Now()
	for i, query := range queries {
		qStart := time.Now()
		s.Search(query, k)
		latencies[i] = time.Since(qStart)
	}
	totalDuration := time.Since(queryStart)
	qps := float64(numQueries) / totalDuration.Seconds()
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	p50, p99, p999 := percentiles(latencies)

	t.Log("\n--- Section 2: Search Performance (SIMD + Sharded + SoA) ---")
	t.Logf("  QPS:           %.2f", qps)
	t.Logf("  P50 Latency:   %v", p50)
	t.Logf("  P99 Latency:   %v", p99)
	t.Logf("  P99.9 Latency: %v", p999)

	// --- Section 3: Memory Layout (AoS vs SoA) ---
	t.Log("\n--- Section 3: Memory Layout (AoS vs SoA) ---")

	// Build AoS store with same data
	rngAoS := rand.New(rand.NewSource(42))
	aos := newAosBenchStore(dimension)
	for i := 0; i < numVectors; i++ {
		aos.insert(fmt.Sprintf("vec-%d", i), generateRandomVector(dimension, rngAoS))
	}

	// Warmup AoS
	for i := 0; i < 10; i++ {
		benchSearchAoS(aos, queries[i], k)
	}

	aosLats := make([]time.Duration, numQueries)
	aosStart := time.Now()
	for i, q := range queries {
		st := time.Now()
		benchSearchAoS(aos, q, k)
		aosLats[i] = time.Since(st)
	}
	aosDur := time.Since(aosStart)
	aosQPS := float64(numQueries) / aosDur.Seconds()
	sort.Slice(aosLats, func(i, j int) bool { return aosLats[i] < aosLats[j] })
	aosP50, aosP99, aosP999 := percentiles(aosLats)

	layoutSpeedup := qps / aosQPS

	t.Logf("  %-8s  QPS: %8.0f  P50: %v  P99: %v  P99.9: %v", "AoS", aosQPS, aosP50, aosP99, aosP999)
	t.Logf("  %-8s  QPS: %8.0f  P50: %v  P99: %v  P99.9: %v", "SoA", qps, p50, p99, p999)
	t.Logf("  SoA Speedup: %.2fx", layoutSpeedup)

	// --- Section 4: Core scalability ---
	t.Log("\n--- Section 4: Core Scalability ---")
	coreConfigs := []int{1, 2, 4, 8}
	maxCores := runtime.NumCPU()

	for _, cores := range coreConfigs {
		if cores > maxCores {
			break
		}
		prev := runtime.GOMAXPROCS(cores)

		// Warmup
		for i := 0; i < 10; i++ {
			s.Search(queries[i], k)
		}

		lats := make([]time.Duration, numQueries)
		qs := time.Now()
		for i, query := range queries {
			st := time.Now()
			s.Search(query, k)
			lats[i] = time.Since(st)
		}
		dur := time.Since(qs)
		coreQPS := float64(numQueries) / dur.Seconds()
		sort.Slice(lats, func(i, j int) bool { return lats[i] < lats[j] })
		cp50, cp99, cp999 := percentiles(lats)

		t.Logf("  GOMAXPROCS=%-2d  QPS: %8.2f  P50: %v  P99: %v  P99.9: %v",
			cores, coreQPS, cp50, cp99, cp999)

		runtime.GOMAXPROCS(prev)
	}

	t.Log("\n============================================================")
	t.Log("                    END OF REPORT")
	t.Log("============================================================")
}

// TestScalability measures search performance at different GOMAXPROCS values.
func TestScalability(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping scalability test in short mode")
	}

	rng := rand.New(rand.NewSource(42))
	s := store.NewVectorStore(dimension)

	for i := 0; i < numVectors; i++ {
		s.Insert(store.Vector{
			ID:   fmt.Sprintf("vec-%d", i),
			Data: generateRandomVector(dimension, rng),
		})
	}

	queries := make([][]float32, numQueries)
	for i := range queries {
		queries[i] = generateRandomVector(dimension, rng)
	}

	t.Log("\n=== Scalability Report ===")
	t.Logf("Vectors: %d, Dimension: %d, k: %d", numVectors, dimension, k)

	coreConfigs := []int{1, 2, 4, 8, 12, 14}
	maxCores := runtime.NumCPU()
	baseQPS := 0.0

	for _, cores := range coreConfigs {
		if cores > maxCores {
			continue
		}
		prev := runtime.GOMAXPROCS(cores)

		// Warmup
		for i := 0; i < 10; i++ {
			s.Search(queries[i], k)
		}

		lats := make([]time.Duration, numQueries)
		qs := time.Now()
		for i, query := range queries {
			st := time.Now()
			s.Search(query, k)
			lats[i] = time.Since(st)
		}
		dur := time.Since(qs)
		coreQPS := float64(numQueries) / dur.Seconds()

		sort.Slice(lats, func(i, j int) bool { return lats[i] < lats[j] })
		p50, p99, p999 := percentiles(lats)

		if cores == 1 {
			baseQPS = coreQPS
		}
		scaling := 0.0
		if baseQPS > 0 {
			scaling = coreQPS / baseQPS
		}

		t.Logf("  Cores=%-2d  QPS: %8.2f  Scaling: %.2fx  P50: %v  P99: %v  P99.9: %v",
			cores, coreQPS, scaling, p50, p99, p999)

		runtime.GOMAXPROCS(prev)
	}
	t.Log("==========================")
}

// aosBenchStore simulates the Week 1 AoS (Array of Structures) layout
// where each vector is a separate []float32 slice (pointer per vector).
type aosBenchStore struct {
	ids  []string
	vecs [][]float32 // each element is a separate heap allocation
	dim  int
}

func newAosBenchStore(dim int) *aosBenchStore {
	return &aosBenchStore{
		ids:  make([]string, 0),
		vecs: make([][]float32, 0),
		dim:  dim,
	}
}

func (s *aosBenchStore) insert(id string, data []float32) {
	s.ids = append(s.ids, id)
	v := make([]float32, len(data))
	copy(v, data)
	s.vecs = append(s.vecs, v)
}

// benchSearchAoS performs brute-force k-NN search over scattered AoS slices.
func benchSearchAoS(s *aosBenchStore, query []float32, k int) []store.SearchResult {
	h := &aosMaxHeap{}
	heap.Init(h)

	for i, vec := range s.vecs {
		dist := distance.EuclideanDistanceSquared(query, vec)
		if h.Len() < k {
			heap.Push(h, store.SearchResult{ID: s.ids[i], Distance: dist})
		} else if dist < (*h)[0].Distance {
			heap.Pop(h)
			heap.Push(h, store.SearchResult{ID: s.ids[i], Distance: dist})
		}
	}

	results := make([]store.SearchResult, h.Len())
	for i := h.Len() - 1; i >= 0; i-- {
		results[i] = heap.Pop(h).(store.SearchResult)
	}
	return results
}

type aosMaxHeap []store.SearchResult

func (h aosMaxHeap) Len() int           { return len(h) }
func (h aosMaxHeap) Less(i, j int) bool { return h[i].Distance > h[j].Distance }
func (h aosMaxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *aosMaxHeap) Push(x any) {
	*h = append(*h, x.(store.SearchResult))
}

func (h *aosMaxHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// TestAoSvsSoA compares search performance between AoS and SoA memory layouts.
func TestAoSvsSoA(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping AoS vs SoA test in short mode")
	}

	rng := rand.New(rand.NewSource(42))

	// Build AoS store
	aos := newAosBenchStore(dimension)
	for i := 0; i < numVectors; i++ {
		aos.insert(fmt.Sprintf("vec-%d", i), generateRandomVector(dimension, rng))
	}

	// Build SoA store (production VectorStore) with same seed
	rng = rand.New(rand.NewSource(42))
	soa := store.NewVectorStore(dimension)
	for i := 0; i < numVectors; i++ {
		soa.Insert(store.Vector{
			ID:   fmt.Sprintf("vec-%d", i),
			Data: generateRandomVector(dimension, rng),
		})
	}

	// Generate queries with a fresh seed
	rng = rand.New(rand.NewSource(99))
	queries := make([][]float32, numQueries)
	for i := range queries {
		queries[i] = generateRandomVector(dimension, rng)
	}

	// Warmup both
	for i := 0; i < 10; i++ {
		benchSearchAoS(aos, queries[i], k)
		soa.Search(queries[i], k)
	}

	// Benchmark AoS
	aosLatencies := make([]time.Duration, numQueries)
	aosStart := time.Now()
	for i, q := range queries {
		s := time.Now()
		benchSearchAoS(aos, q, k)
		aosLatencies[i] = time.Since(s)
	}
	aosDuration := time.Since(aosStart)
	aosQPS := float64(numQueries) / aosDuration.Seconds()

	sort.Slice(aosLatencies, func(i, j int) bool { return aosLatencies[i] < aosLatencies[j] })
	aosP50, aosP99, aosP999 := percentiles(aosLatencies)

	// Benchmark SoA
	soaLatencies := make([]time.Duration, numQueries)
	soaStart := time.Now()
	for i, q := range queries {
		s := time.Now()
		soa.Search(q, k)
		soaLatencies[i] = time.Since(s)
	}
	soaDuration := time.Since(soaStart)
	soaQPS := float64(numQueries) / soaDuration.Seconds()

	sort.Slice(soaLatencies, func(i, j int) bool { return soaLatencies[i] < soaLatencies[j] })
	soaP50, soaP99, soaP999 := percentiles(soaLatencies)

	speedup := soaQPS / aosQPS

	t.Log("\n=== Memory Layout: AoS vs SoA ===")
	t.Logf("Vectors: %d, Dimension: %d, k: %d", numVectors, dimension, k)
	t.Logf("GOMAXPROCS: %d", runtime.GOMAXPROCS(0))
	t.Log("--------------------------------------------------")
	t.Logf("%-8s  %8s  %12s  %12s  %12s", "Layout", "QPS", "P50", "P99", "P99.9")
	t.Logf("%-8s  %8.0f  %12v  %12v  %12v", "AoS", aosQPS, aosP50, aosP99, aosP999)
	t.Logf("%-8s  %8.0f  %12v  %12v  %12v", "SoA", soaQPS, soaP50, soaP99, soaP999)
	t.Logf("--------------------------------------------------")
	t.Logf("SoA Speedup: %.2fx", speedup)
	t.Log("==================================================")
}

func percentiles(sorted []time.Duration) (p50, p99, p999 time.Duration) {
	n := len(sorted)
	p50 = sorted[n/2]
	p99 = sorted[int(float64(n)*0.99)]
	idx999 := int(float64(n) * 0.999)
	if idx999 >= n {
		idx999 = n - 1
	}
	p999 = sorted[idx999]
	return
}
