package distance

import (
	"math"
	"math/rand"
	"testing"
)

func generateVector(dim int, rng *rand.Rand) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}
	return v
}

// TestEuclideanDistanceSquaredCorrectness verifies NEON matches scalar at various dimensions.
func TestEuclideanDistanceSquaredCorrectness(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	dims := []int{4, 8, 16, 32, 64, 128, 256}

	for _, dim := range dims {
		a := generateVector(dim, rng)
		b := generateVector(dim, rng)

		got := EuclideanDistanceSquared(a, b)
		want := EuclideanDistanceSquaredScalar(a, b)

		rel := relError(got, want)
		if rel > 1e-5 {
			t.Errorf("dim=%d: EuclideanDistanceSquared=%v, scalar=%v (rel error=%v)", dim, got, want, rel)
		}
	}
}

// TestDotProductCorrectness verifies NEON matches scalar at various dimensions.
func TestDotProductCorrectness(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	dims := []int{4, 8, 16, 32, 64, 128, 256}

	for _, dim := range dims {
		a := generateVector(dim, rng)
		b := generateVector(dim, rng)

		got := DotProduct(a, b)
		want := DotProductScalar(a, b)

		rel := relError(got, want)
		if rel > 1e-5 {
			t.Errorf("dim=%d: DotProduct=%v, scalar=%v (rel error=%v)", dim, got, want, rel)
		}
	}
}

// TestMagnitudeUsesNEON verifies Magnitude benefits from NEON via DotProduct.
func TestMagnitudeUsesNEON(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	v := generateVector(128, rng)

	got := Magnitude(v)
	// Compute scalar magnitude
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	want := float32(math.Sqrt(float64(sum)))

	rel := relError(got, want)
	if rel > 1e-5 {
		t.Errorf("Magnitude=%v, expected=%v (rel error=%v)", got, want, rel)
	}
}

// TestCosineSimilarityCorrectness verifies cosine similarity with NEON.
func TestCosineSimilarityCorrectness(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	a := generateVector(128, rng)
	b := generateVector(128, rng)

	got := CosineSimilarity(a, b)
	// Manual scalar computation
	dot := DotProductScalar(a, b)
	var sumA, sumB float32
	for i := range a {
		sumA += a[i] * a[i]
		sumB += b[i] * b[i]
	}
	magA := float32(math.Sqrt(float64(sumA)))
	magB := float32(math.Sqrt(float64(sumB)))
	want := dot / (magA * magB)

	rel := relError(got, want)
	if rel > 1e-5 {
		t.Errorf("CosineSimilarity=%v, expected=%v (rel error=%v)", got, want, rel)
	}
}

// TestSmallVectors tests edge cases with very small vectors.
func TestSmallVectors(t *testing.T) {
	// Length 1 (scalar fallback)
	a := []float32{3.0}
	b := []float32{1.0}
	if got, want := EuclideanDistanceSquared(a, b), float32(4.0); got != want {
		t.Errorf("EuclideanDistanceSquared([3],[1])=%v, want %v", got, want)
	}
	if got, want := DotProduct(a, b), float32(3.0); got != want {
		t.Errorf("DotProduct([3],[1])=%v, want %v", got, want)
	}

	// Length 4 (minimum NEON path)
	a4 := []float32{1, 2, 3, 4}
	b4 := []float32{5, 6, 7, 8}
	gotDot := DotProduct(a4, b4)
	wantDot := float32(1*5 + 2*6 + 3*7 + 4*8) // 70
	if gotDot != wantDot {
		t.Errorf("DotProduct 4-elem=%v, want %v", gotDot, wantDot)
	}

	gotEuc := EuclideanDistanceSquared(a4, b4)
	wantEuc := float32(4*4 + 4*4 + 4*4 + 4*4) // 64
	if gotEuc != wantEuc {
		t.Errorf("EuclideanDistanceSquared 4-elem=%v, want %v", gotEuc, wantEuc)
	}
}

// TestZeroVectors ensures zero vectors return zero distance/product.
func TestZeroVectors(t *testing.T) {
	a := make([]float32, 128)
	b := make([]float32, 128)

	if got := EuclideanDistanceSquared(a, b); got != 0 {
		t.Errorf("EuclideanDistanceSquared(zero, zero)=%v, want 0", got)
	}
	if got := DotProduct(a, b); got != 0 {
		t.Errorf("DotProduct(zero, zero)=%v, want 0", got)
	}
}

func relError(got, want float32) float64 {
	if want == 0 {
		return float64(math.Abs(float64(got)))
	}
	return math.Abs(float64(got-want)) / math.Abs(float64(want))
}

// --- Benchmarks ---

func BenchmarkDotProductScalar(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	a := generateVector(128, rng)
	v := generateVector(128, rng)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DotProductScalar(a, v)
	}
}

func BenchmarkDotProductNEON(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	a := generateVector(128, rng)
	v := generateVector(128, rng)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DotProduct(a, v)
	}
}

func BenchmarkEuclideanDistanceSquaredScalar(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	a := generateVector(128, rng)
	v := generateVector(128, rng)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		EuclideanDistanceSquaredScalar(a, v)
	}
}

func BenchmarkEuclideanDistanceSquaredNEON(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	a := generateVector(128, rng)
	v := generateVector(128, rng)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		EuclideanDistanceSquared(a, v)
	}
}
