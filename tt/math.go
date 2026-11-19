package tt

import "math"

// Element math. These functions are what element functions passed to Map may
// call. tessc maps each one to a vector-unit (SFPU) operation on the card; on
// the CPU they are computed in float64 and rounded to float32.

// Exp returns e**x.
func Exp(x float32) float32 { return float32(math.Exp(float64(x))) }

// Log returns the natural logarithm of x.
func Log(x float32) float32 { return float32(math.Log(float64(x))) }

// Sqrt returns the square root of x.
func Sqrt(x float32) float32 { return float32(math.Sqrt(float64(x))) }

// Rsqrt returns 1/sqrt(x).
func Rsqrt(x float32) float32 { return float32(1 / math.Sqrt(float64(x))) }

// Recip returns 1/x.
func Recip(x float32) float32 { return 1 / x }

// Abs returns |x|.
func Abs(x float32) float32 { return float32(math.Abs(float64(x))) }

// Relu returns max(x, 0).
func Relu(x float32) float32 {
	if x > 0 {
		return x
	}
	return 0
}

// Sigmoid returns 1/(1+e**-x).
func Sigmoid(x float32) float32 { return float32(1 / (1 + math.Exp(-float64(x)))) }

// Tanh returns the hyperbolic tangent of x.
func Tanh(x float32) float32 { return float32(math.Tanh(float64(x))) }

// Sin returns the sine of x (radians).
func Sin(x float32) float32 { return float32(math.Sin(float64(x))) }

// Cos returns the cosine of x (radians).
func Cos(x float32) float32 { return float32(math.Cos(float64(x))) }

// Square returns x*x.
func Square(x float32) float32 { return x * x }

// Max returns the larger of a and b.
func Max(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

// Min returns the smaller of a and b.
func Min(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}
