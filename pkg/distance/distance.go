package distance

import "math"

// EuclideanDistanceSquared computes the squared Euclidean distance between two vectors.
// On arm64, this dispatches to NEON-accelerated assembly for vectors with len >= 4.
func EuclideanDistanceSquared(a, b []float32) float32 {
	return euclideanDistanceSquaredPlatform(a, b)
}

// EuclideanDistanceSquaredScalar is the pure-Go scalar implementation.
// Exported for benchmarking comparisons.
func EuclideanDistanceSquaredScalar(a, b []float32) float32 {
	var sum float32
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

// EuclideanDistance computes the Euclidean (L2) distance between two vectors.
func EuclideanDistance(a, b []float32) float32 {
	return float32(math.Sqrt(float64(EuclideanDistanceSquared(a, b))))
}

// DotProduct computes the dot product of two vectors.
// On arm64, this dispatches to NEON-accelerated assembly for vectors with len >= 4.
func DotProduct(a, b []float32) float32 {
	return dotProductPlatform(a, b)
}

// DotProductScalar is the pure-Go scalar implementation.
// Exported for benchmarking comparisons.
func DotProductScalar(a, b []float32) float32 {
	var sum float32
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// Magnitude computes the L2 norm (magnitude) of a vector.
// Uses DotProduct(v,v) to benefit from NEON acceleration.
func Magnitude(v []float32) float32 {
	return float32(math.Sqrt(float64(DotProduct(v, v))))
}

// CosineSimilarity computes the cosine similarity between two vectors.
func CosineSimilarity(a, b []float32) float32 {
	dot := DotProduct(a, b)
	magA := Magnitude(a)
	magB := Magnitude(b)
	if magA == 0 || magB == 0 {
		return 0
	}
	return dot / (magA * magB)
}

// CosineDistance converts cosine similarity to a distance metric (1 - similarity).
func CosineDistance(a, b []float32) float32 {
	return 1 - CosineSimilarity(a, b)
}
