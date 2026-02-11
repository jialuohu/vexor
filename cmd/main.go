package main

import (
	"fmt"
	"math/rand"
	"time"

	"vexor/pkg/store"
)

func main() {
	const (
		numVectors = 10_000
		dimension  = 128
		k          = 5
	)

	fmt.Println("Vexor - Vector Similarity Search Engine")
	fmt.Println("========================================")
	fmt.Printf("Initializing store with dimension %d\n", dimension)

	s := store.NewVectorStore(dimension)
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Insert vectors
	fmt.Printf("Inserting %d vectors...\n", numVectors)
	start := time.Now()
	for i := range numVectors {
		v := make([]float32, dimension)
		for j := range v {
			v[j] = rng.Float32()*2 - 1
		}
		s.Insert(store.Vector{
			ID:   fmt.Sprintf("vec-%d", i),
			Data: v,
		})
	}
	fmt.Printf("Inserted %d vectors in %v\n", numVectors, time.Since(start))

	// Run a sample query
	query := make([]float32, dimension)
	for i := range query {
		query[i] = rng.Float32()*2 - 1
	}

	fmt.Printf("\nSearching for %d nearest neighbors...\n", k)
	start = time.Now()
	results, err := s.Search(query, k)
	if err != nil {
		fmt.Printf("Search error: %v\n", err)
		return
	}
	fmt.Printf("Search completed in %v\n\n", time.Since(start))

	fmt.Println("Top results:")
	for i, r := range results {
		fmt.Printf("  %d. %s (distance: %.4f)\n", i+1, r.ID, r.Distance)
	}

	fmt.Println("\nRun 'go test -v ./bench/' for full benchmark suite")
}
