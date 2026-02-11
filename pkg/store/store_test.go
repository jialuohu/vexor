package store

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"
)

func TestInsertAndCount(t *testing.T) {
	s := NewVectorStore(3)
	if s.Count() != 0 {
		t.Fatalf("expected 0, got %d", s.Count())
	}

	s.Insert(Vector{ID: "a", Data: []float32{1, 2, 3}})
	s.Insert(Vector{ID: "b", Data: []float32{4, 5, 6}})
	if s.Count() != 2 {
		t.Fatalf("expected 2, got %d", s.Count())
	}
}

func TestInsertUpdate(t *testing.T) {
	s := NewVectorStore(2)
	s.Insert(Vector{ID: "a", Data: []float32{1, 2}})
	s.Insert(Vector{ID: "a", Data: []float32{3, 4}}) // update

	if s.Count() != 1 {
		t.Fatalf("expected 1 after update, got %d", s.Count())
	}

	results, _ := s.Search([]float32{3, 4}, 1)
	if len(results) != 1 || results[0].ID != "a" || results[0].Distance != 0 {
		t.Fatalf("unexpected result after update: %+v", results)
	}
}

func TestInsertErrors(t *testing.T) {
	s := NewVectorStore(3)
	if err := s.Insert(Vector{ID: "", Data: []float32{1, 2, 3}}); err != ErrEmptyID {
		t.Fatalf("expected ErrEmptyID, got %v", err)
	}
	if err := s.Insert(Vector{ID: "a", Data: []float32{1, 2}}); err != ErrDimensionMismatch {
		t.Fatalf("expected ErrDimensionMismatch, got %v", err)
	}
}

func TestDelete(t *testing.T) {
	s := NewVectorStore(2)
	s.Insert(Vector{ID: "a", Data: []float32{1, 2}})
	s.Insert(Vector{ID: "b", Data: []float32{3, 4}})
	s.Insert(Vector{ID: "c", Data: []float32{5, 6}})

	if err := s.Delete("b"); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	if s.Count() != 2 {
		t.Fatalf("expected 2 after delete, got %d", s.Count())
	}
	if err := s.Delete("nonexistent"); err != ErrNotFound {
		t.Fatalf("expected ErrNotFound, got %v", err)
	}
}

func TestSearchBasic(t *testing.T) {
	s := NewVectorStore(2)
	s.Insert(Vector{ID: "origin", Data: []float32{0, 0}})
	s.Insert(Vector{ID: "near", Data: []float32{1, 0}})
	s.Insert(Vector{ID: "far", Data: []float32{10, 10}})

	results, err := s.Search([]float32{0, 0}, 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].ID != "origin" {
		t.Errorf("expected 'origin' as nearest, got %q", results[0].ID)
	}
	if results[1].ID != "near" {
		t.Errorf("expected 'near' as second, got %q", results[1].ID)
	}
}

func TestSearchCosineBasic(t *testing.T) {
	s := NewVectorStore(2)
	s.Insert(Vector{ID: "same_dir", Data: []float32{1, 0}})
	s.Insert(Vector{ID: "perp", Data: []float32{0, 1}})
	s.Insert(Vector{ID: "opposite", Data: []float32{-1, 0}})

	results, err := s.SearchCosine([]float32{1, 0}, 1)
	if err != nil {
		t.Fatalf("SearchCosine failed: %v", err)
	}
	if results[0].ID != "same_dir" {
		t.Errorf("expected 'same_dir', got %q", results[0].ID)
	}
}

func TestSearchEdgeCases(t *testing.T) {
	s := NewVectorStore(2)

	// Empty store
	results, err := s.Search([]float32{1, 2}, 5)
	if err != nil {
		t.Fatalf("Search on empty store failed: %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("expected 0 results on empty store")
	}

	// k=0
	s.Insert(Vector{ID: "a", Data: []float32{1, 2}})
	results, err = s.Search([]float32{1, 2}, 0)
	if err != nil {
		t.Fatalf("Search k=0 failed: %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("expected 0 results for k=0")
	}

	// Dimension mismatch
	_, err = s.Search([]float32{1, 2, 3}, 1)
	if err != ErrDimensionMismatch {
		t.Fatalf("expected ErrDimensionMismatch, got %v", err)
	}
}

func TestDimension(t *testing.T) {
	s := NewVectorStore(42)
	if s.Dimension() != 42 {
		t.Fatalf("expected 42, got %d", s.Dimension())
	}
}

func TestSearchReturnsCorrectDistances(t *testing.T) {
	s := NewVectorStore(2)
	s.Insert(Vector{ID: "a", Data: []float32{3, 4}})

	results, _ := s.Search([]float32{0, 0}, 1)
	expected := float32(math.Sqrt(9 + 16)) // 5.0
	if math.Abs(float64(results[0].Distance-expected)) > 1e-5 {
		t.Errorf("expected distance %v, got %v", expected, results[0].Distance)
	}
}

// TestShardDistribution verifies vectors distribute across shards.
func TestShardDistribution(t *testing.T) {
	s := NewVectorStore(2)
	n := 10000
	for i := 0; i < n; i++ {
		s.Insert(Vector{ID: fmt.Sprintf("v-%d", i), Data: []float32{float32(i), float32(i)}})
	}
	if s.Count() != n {
		t.Fatalf("expected %d, got %d", n, s.Count())
	}

	// Check that vectors are distributed (not all in one shard)
	maxPerShard := 0
	minPerShard := n
	for i := range s.shards {
		c := len(s.shards[i].ids)
		if c > maxPerShard {
			maxPerShard = c
		}
		if c < minPerShard {
			minPerShard = c
		}
	}
	// With FNV hash and 10k vectors, expect reasonable distribution
	if minPerShard == 0 {
		t.Error("at least one shard is empty — poor distribution")
	}
	t.Logf("Shard distribution: min=%d, max=%d (of %d total)", minPerShard, maxPerShard, n)
}

// TestConcurrentInsertDelete stress-tests concurrent inserts and deletes.
func TestConcurrentInsertDelete(t *testing.T) {
	s := NewVectorStore(8)
	rng := rand.New(rand.NewSource(42))

	// Pre-generate vectors
	vectors := make([]Vector, 1000)
	for i := range vectors {
		data := make([]float32, 8)
		for j := range data {
			data[j] = rng.Float32()
		}
		vectors[i] = Vector{ID: fmt.Sprintf("v-%d", i), Data: data}
	}

	// Concurrent inserts
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			for j := start; j < start+100; j++ {
				s.Insert(vectors[j])
			}
		}(i * 100)
	}
	wg.Wait()

	if s.Count() != 1000 {
		t.Fatalf("expected 1000 after concurrent insert, got %d", s.Count())
	}

	// Concurrent deletes
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			for j := start; j < start+50; j++ {
				s.Delete(fmt.Sprintf("v-%d", j))
			}
		}(i * 100)
	}
	wg.Wait()

	if s.Count() != 500 {
		t.Fatalf("expected 500 after concurrent delete, got %d", s.Count())
	}
}

// TestConcurrentInsertSearch stress-tests concurrent inserts and searches.
func TestConcurrentInsertSearch(t *testing.T) {
	s := NewVectorStore(8)
	rng := rand.New(rand.NewSource(42))

	// Insert some initial data
	for i := 0; i < 500; i++ {
		data := make([]float32, 8)
		for j := range data {
			data[j] = rng.Float32()
		}
		s.Insert(Vector{ID: fmt.Sprintf("v-%d", i), Data: data})
	}

	var wg sync.WaitGroup
	// Writers
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			localRng := rand.New(rand.NewSource(int64(start)))
			for j := 0; j < 100; j++ {
				data := make([]float32, 8)
				for k := range data {
					data[k] = localRng.Float32()
				}
				s.Insert(Vector{ID: fmt.Sprintf("w-%d-%d", start, j), Data: data})
			}
		}(i)
	}
	// Readers
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			localRng := rand.New(rand.NewSource(int64(seed)))
			for j := 0; j < 50; j++ {
				query := make([]float32, 8)
				for k := range query {
					query[k] = localRng.Float32()
				}
				results, err := s.Search(query, 5)
				if err != nil {
					t.Errorf("Search failed: %v", err)
					return
				}
				if len(results) == 0 {
					t.Error("expected non-empty results")
					return
				}
			}
		}(i + 100)
	}
	wg.Wait()
}
