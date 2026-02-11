//go:build !arm64

package distance

func euclideanDistanceSquaredPlatform(a, b []float32) float32 {
	return EuclideanDistanceSquaredScalar(a, b)
}

func dotProductPlatform(a, b []float32) float32 {
	return DotProductScalar(a, b)
}
