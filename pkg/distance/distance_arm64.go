//go:build arm64

package distance

//go:noescape
func euclideanDistanceSquaredNEON(a, b []float32) float32

//go:noescape
func dotProductNEON(a, b []float32) float32

func euclideanDistanceSquaredPlatform(a, b []float32) float32 {
	if len(a) >= 4 {
		return euclideanDistanceSquaredNEON(a, b)
	}
	return EuclideanDistanceSquaredScalar(a, b)
}

func dotProductPlatform(a, b []float32) float32 {
	if len(a) >= 4 {
		return dotProductNEON(a, b)
	}
	return DotProductScalar(a, b)
}
