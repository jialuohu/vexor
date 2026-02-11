package store

import (
	"container/heap"
	"errors"
	"hash/fnv"
	"math"
	"runtime"
	"sync"

	"vexor/pkg/distance"
)

var (
	ErrDimensionMismatch = errors.New("vector dimension does not match store dimension")
	ErrEmptyID           = errors.New("vector ID cannot be empty")
	ErrNotFound          = errors.New("vector not found")
)

const numShards = 16

// Vector represents a vector with an ID and float32 data.
type Vector struct {
	ID   string
	Data []float32
}

// SearchResult represents a search result with distance information.
type SearchResult struct {
	ID       string
	Distance float32
}

// shard uses SoA (Structure of Arrays) layout for cache-friendly access.
// Vector i's data lives at data[i*dim : (i+1)*dim] in a contiguous allocation.
type shard struct {
	ids     []string
	data    []float32 // contiguous: vector i at data[i*dim : (i+1)*dim]
	idIndex map[string]int
	mu      sync.RWMutex
}

// VectorStore is an in-memory store for vectors supporting k-NN search.
// Uses 16 shards with per-shard locks and SoA memory layout.
type VectorStore struct {
	shards    [numShards]shard
	dimension int
}

// NewVectorStore creates a new VectorStore with the specified dimension.
func NewVectorStore(dimension int) *VectorStore {
	vs := &VectorStore{dimension: dimension}
	for i := range vs.shards {
		vs.shards[i].ids = make([]string, 0)
		vs.shards[i].data = make([]float32, 0)
		vs.shards[i].idIndex = make(map[string]int)
	}
	return vs
}

func shardIndex(id string) int {
	h := fnv.New32a()
	h.Write([]byte(id))
	return int(h.Sum32() % numShards)
}

// Insert adds a vector to the store.
func (s *VectorStore) Insert(v Vector) error {
	if v.ID == "" {
		return ErrEmptyID
	}
	if len(v.Data) != s.dimension {
		return ErrDimensionMismatch
	}

	sh := &s.shards[shardIndex(v.ID)]
	sh.mu.Lock()
	defer sh.mu.Unlock()

	dim := s.dimension

	if idx, exists := sh.idIndex[v.ID]; exists {
		// Update existing: copy new data into the contiguous slice
		copy(sh.data[idx*dim:(idx+1)*dim], v.Data)
		return nil
	}

	sh.idIndex[v.ID] = len(sh.ids)
	sh.ids = append(sh.ids, v.ID)
	sh.data = append(sh.data, v.Data...)
	return nil
}

// Delete removes a vector from the store by ID.
func (s *VectorStore) Delete(id string) error {
	sh := &s.shards[shardIndex(id)]
	sh.mu.Lock()
	defer sh.mu.Unlock()

	idx, exists := sh.idIndex[id]
	if !exists {
		return ErrNotFound
	}

	dim := s.dimension
	lastIdx := len(sh.ids) - 1

	if idx != lastIdx {
		// Swap with last: copy last vector's data into the deleted slot
		sh.ids[idx] = sh.ids[lastIdx]
		copy(sh.data[idx*dim:(idx+1)*dim], sh.data[lastIdx*dim:(lastIdx+1)*dim])
		sh.idIndex[sh.ids[idx]] = idx
	}

	sh.ids = sh.ids[:lastIdx]
	sh.data = sh.data[:lastIdx*dim]
	delete(sh.idIndex, id)

	return nil
}

// Count returns the number of vectors in the store.
func (s *VectorStore) Count() int {
	total := 0
	for i := range s.shards {
		s.shards[i].mu.RLock()
		total += len(s.shards[i].ids)
		s.shards[i].mu.RUnlock()
	}
	return total
}

// Dimension returns the dimension of vectors in this store.
func (s *VectorStore) Dimension() int {
	return s.dimension
}

// Search performs a k-NN search using Euclidean distance.
// Parallelizes across shards using multiple goroutines.
func (s *VectorStore) Search(query []float32, k int) ([]SearchResult, error) {
	if len(query) != s.dimension {
		return nil, ErrDimensionMismatch
	}
	if k <= 0 {
		return []SearchResult{}, nil
	}

	dim := s.dimension
	nWorkers := runtime.GOMAXPROCS(0)
	if nWorkers > numShards {
		nWorkers = numShards
	}

	type workerResult struct {
		results []SearchResult
	}
	workerResults := make([]workerResult, nWorkers)

	var wg sync.WaitGroup
	shardsPerWorker := (numShards + nWorkers - 1) / nWorkers

	for w := 0; w < nWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			h := &maxHeap{}
			heap.Init(h)

			start := workerID * shardsPerWorker
			end := start + shardsPerWorker
			if end > numShards {
				end = numShards
			}

			for si := start; si < end; si++ {
				sh := &s.shards[si]
				sh.mu.RLock()
				n := len(sh.ids)
				for i := 0; i < n; i++ {
					vec := sh.data[i*dim : (i+1)*dim]
					dist := distance.EuclideanDistanceSquared(query, vec)
					if h.Len() < k {
						heap.Push(h, SearchResult{ID: sh.ids[i], Distance: dist})
					} else if dist < (*h)[0].Distance {
						heap.Pop(h)
						heap.Push(h, SearchResult{ID: sh.ids[i], Distance: dist})
					}
				}
				sh.mu.RUnlock()
			}

			results := make([]SearchResult, h.Len())
			for i := h.Len() - 1; i >= 0; i-- {
				results[i] = heap.Pop(h).(SearchResult)
			}
			workerResults[workerID] = workerResult{results: results}
		}(w)
	}
	wg.Wait()

	// Merge all worker results into final top-k
	finalHeap := &maxHeap{}
	heap.Init(finalHeap)
	for _, wr := range workerResults {
		for _, r := range wr.results {
			if finalHeap.Len() < k {
				heap.Push(finalHeap, r)
			} else if r.Distance < (*finalHeap)[0].Distance {
				heap.Pop(finalHeap)
				heap.Push(finalHeap, r)
			}
		}
	}

	results := make([]SearchResult, finalHeap.Len())
	for i := finalHeap.Len() - 1; i >= 0; i-- {
		r := heap.Pop(finalHeap).(SearchResult)
		r.Distance = sqrt32(r.Distance)
		results[i] = r
	}

	return results, nil
}

// SearchCosine performs a k-NN search using cosine distance.
func (s *VectorStore) SearchCosine(query []float32, k int) ([]SearchResult, error) {
	if len(query) != s.dimension {
		return nil, ErrDimensionMismatch
	}
	if k <= 0 {
		return []SearchResult{}, nil
	}

	dim := s.dimension
	nWorkers := runtime.GOMAXPROCS(0)
	if nWorkers > numShards {
		nWorkers = numShards
	}

	type workerResult struct {
		results []SearchResult
	}
	workerResults := make([]workerResult, nWorkers)

	var wg sync.WaitGroup
	shardsPerWorker := (numShards + nWorkers - 1) / nWorkers

	for w := 0; w < nWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			h := &maxHeap{}
			heap.Init(h)

			start := workerID * shardsPerWorker
			end := start + shardsPerWorker
			if end > numShards {
				end = numShards
			}

			for si := start; si < end; si++ {
				sh := &s.shards[si]
				sh.mu.RLock()
				n := len(sh.ids)
				for i := 0; i < n; i++ {
					vec := sh.data[i*dim : (i+1)*dim]
					dist := distance.CosineDistance(query, vec)
					if h.Len() < k {
						heap.Push(h, SearchResult{ID: sh.ids[i], Distance: dist})
					} else if dist < (*h)[0].Distance {
						heap.Pop(h)
						heap.Push(h, SearchResult{ID: sh.ids[i], Distance: dist})
					}
				}
				sh.mu.RUnlock()
			}

			results := make([]SearchResult, h.Len())
			for i := h.Len() - 1; i >= 0; i-- {
				results[i] = heap.Pop(h).(SearchResult)
			}
			workerResults[workerID] = workerResult{results: results}
		}(w)
	}
	wg.Wait()

	finalHeap := &maxHeap{}
	heap.Init(finalHeap)
	for _, wr := range workerResults {
		for _, r := range wr.results {
			if finalHeap.Len() < k {
				heap.Push(finalHeap, r)
			} else if r.Distance < (*finalHeap)[0].Distance {
				heap.Pop(finalHeap)
				heap.Push(finalHeap, r)
			}
		}
	}

	results := make([]SearchResult, finalHeap.Len())
	for i := finalHeap.Len() - 1; i >= 0; i-- {
		results[i] = heap.Pop(finalHeap).(SearchResult)
	}

	return results, nil
}

func sqrt32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

// maxHeap implements heap.Interface for SearchResult (max-heap by distance).
type maxHeap []SearchResult

func (h maxHeap) Len() int           { return len(h) }
func (h maxHeap) Less(i, j int) bool { return h[i].Distance > h[j].Distance }
func (h maxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *maxHeap) Push(x any) {
	*h = append(*h, x.(SearchResult))
}

func (h *maxHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}
