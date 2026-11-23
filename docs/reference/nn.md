---
title: Package tt/nn
parent: Reference
nav_order: 3
---

# Package tt/nn

```go
import "github.com/pavelgladyshev/tessera/tt/nn"
```

This page is made from the package's doc comments by `tools/apidoc`.
{: .fs-2 }

Package nn runs whole-tensor operations from a Tessera program's main:
on the card through ttnn, Tenstorrent's library of neural-network
operations, and on the CPU (go run, without a card) in plain float32.

	a := nn.FromFloats([]int{64, 128}, values) // host -> card
	b := nn.MatMul(a, w)                       // runs on the card
	out := nn.ReLU(b).Floats()                 // card -> host

A Tensor made on the card lives there: Go holds a handle, and the numbers
travel only with FromFloats and Floats. Every operation returns when it
has run. A mistake (shapes that do not fit, a tensor used after Free, an
error inside ttnn) is a panic that says which operation it was.

On the card the values are bf16 (8 bits of mantissa), and ttnn's
functions are approximations of their own, so results differ from the
CPU's in the last places.

nn is for main only: worker code (the function given to tt.Parallel)
uses tt.Tensor instead. An nn call waits for a tt.Parallel that is
running: the two take turns on the card.

## Functions

### ArgMax

```go
func ArgMax(a Tensor) []int
```

ArgMax returns, for each line of a along its last dimension, the index of
its largest element: for an m × n tensor, m indices. Unlike the other
operations it returns numbers to main. Where elements are equal or
nearly so, the card can choose differently from the CPU.

## Types

### type Conv

```go
type Conv struct {
	Stride		[2]int	// down and across; 0 is 1
	Padding		[2]int	// zeros added above and below, and left and right
	Dilation	[2]int	// distance between the kernel's elements; 0 is 1
	Groups		int	// 0 is 1
}
```

Conv says how Conv2d moves its kernel. The zero Conv is stride 1, no
padding, no dilation, one group.

### type Format

```go
type Format int
```

Format is how a tensor's numbers are kept on the card.

### type Pool

```go
type Pool struct {
	Kernel	[2]int	// the window, down and across
	Stride	[2]int	// 0 is the window's size
	Padding	[2]int
}
```

Pool says how MaxPool2d and AvgPool2d move their window.

### type Tensor

```go
type Tensor struct {
	// contains filtered or unexported fields
}
```

Tensor is a tensor of 1 to 4 dimensions. The zero Tensor is not usable.

#### Add

```go
func Add(a, b Tensor) Tensor
```

Add returns a + b, element by element. The shapes are the same, or one
broadcasts to the other: lined up at their last dimensions, a dimension
of 1 stretches (a tensor of shape [n] adds to every row of an m × n one).
Sub, Mul and Div do the same.

#### Attention

```go
func Attention(q, k, v Tensor, causal bool) Tensor
```

Attention is scaled dot-product attention: softmax(q·kᵀ/√d)·v, where q
is [... s d], k is [... t d] and v is [... t e], with the same leading
dimensions; the result is [... s e]. With causal, position i of q sees
only positions 0 to i of k and v (and s must equal t).

#### AvgPool2d

```go
func AvgPool2d(x Tensor, p Pool) Tensor
```

AvgPool2d returns the mean of each window of x, [n h w c]. Padding
counts as zeros.

#### Concat

```go
func Concat(d int, ts ...Tensor) Tensor
```

Concat joins tensors along dimension d: their other dimensions are the
same.

#### Conv2d

```go
func Conv2d(x, w, b Tensor, c Conv) Tensor
```

Conv2d is a 2-D convolution. x is [n h w c] (channels last, as ttnn has
them), w is [out c/groups kh kw] (as PyTorch has them) and b has shape
[out], or is the zero Tensor for no biases. The result is [n h' w' out].

#### Div

```go
func Div(a, b Tensor) Tensor
```

Div returns a / b, element by element.

#### Embedding

```go
func Embedding(ids []int, table Tensor) Tensor
```

Embedding returns the rows of table, rows × n, that ids name: a tensor
of shape [len(ids) n].

#### Exp

```go
func Exp(a Tensor) Tensor
```

Exp returns e to the power of each element.

#### FromFloats

```go
func FromFloats(shape []int, v []float32) Tensor
```

FromFloats makes a tensor of that shape from v, whose last dimension
runs fastest: element (i, j) of a rows × cols tensor is v[i*cols+j].

#### FromFloatsAs

```go
func FromFloatsAs(f Format, shape []int, v []float32) Tensor
```

FromFloatsAs is FromFloats in a format of the card's. Use the result as
a weight (the second operand of MatMul or Linear, say); what operations
give is BF16 again. Under go run the numbers are rounded as the card
would round them, nearly: the card's own rounding may differ in the last
bit.

#### Full

```go
func Full(shape []int, value float32) Tensor
```

Full makes a tensor of that shape with every element equal to value.

#### GELU

```go
func GELU(a Tensor) Tensor
```

GELU returns x·Φ(x) for each element x, Φ being the normal distribution's
cumulative function.

#### LayerNorm

```go
func LayerNorm(a, gamma, beta Tensor, eps float32) Tensor
```

LayerNorm normalizes a along its last dimension, of size n: each line
becomes (x - mean) / sqrt(variance + eps) * gamma + beta, where gamma and
beta are tensors of shape [n].

#### Linear

```go
func Linear(a, w, b Tensor) Tensor
```

Linear returns a·w + b: a is m × k, w is k × n and b has shape [n]. b can
be the zero Tensor, for no biases.

#### Log

```go
func Log(a Tensor) Tensor
```

Log returns the natural logarithm of each element.

#### MatMul

```go
func MatMul(a, b Tensor) Tensor
```

MatMul returns the matrix product of a (m × k) and b (k × n). With more
dimensions, the ones before the last two are the same in a and b, and
the result is the product of each pair of matrices.

#### Max

```go
func Max(a Tensor, d int) Tensor
```

Max is the largest element of a along dimension d; the result's shape is
as Sum's.

#### MaxPool2d

```go
func MaxPool2d(x Tensor, p Pool) Tensor
```

MaxPool2d returns the largest element of each window of x, [n h w c].

#### Mean

```go
func Mean(a Tensor, d int) Tensor
```

Mean is the mean of a along dimension d; the result's shape is as Sum's.

#### Mul

```go
func Mul(a, b Tensor) Tensor
```

Mul returns a * b, element by element (MatMul is the matrix product).

#### Ones

```go
func Ones(shape []int) Tensor
```

Ones makes a tensor of ones.

#### PReLU

```go
func PReLU(a, slope Tensor) Tensor
```

PReLU returns a where it is positive and slope·a where it is negative.
slope has shape [c], one for each index of a's last dimension (the
channels of an [n h w c] image), or [1] for one slope for all.

#### RMSNorm

```go
func RMSNorm(a, gamma Tensor, eps float32) Tensor
```

RMSNorm scales each line of a along its last dimension, of size n, to a
root mean square of 1 and multiplies by gamma, of shape [n]:
x / sqrt(mean(x²) + eps) * gamma.

#### ReLU

```go
func ReLU(a Tensor) Tensor
```

ReLU returns max(0, a), element by element.

#### Reshape

```go
func Reshape(a Tensor, shape []int) Tensor
```

Reshape returns a's elements, in the same order, as a tensor of another
shape with as many elements.

#### SiLU

```go
func SiLU(a Tensor) Tensor
```

SiLU returns a·sigmoid(a), element by element.

#### Sigmoid

```go
func Sigmoid(a Tensor) Tensor
```

Sigmoid returns 1/(1+exp(-a)), element by element.

#### Slice

```go
func Slice(a Tensor, begin, end []int) Tensor
```

Slice returns the part of a from index begin[i] up to, not including,
end[i] in each dimension i.

#### Softmax

```go
func Softmax(a Tensor, d int) Tensor
```

Softmax returns exp(a) over its sum along dimension d: along d, the
result's elements are positive and add up to 1.

#### Sqrt

```go
func Sqrt(a Tensor) Tensor
```

Sqrt returns the square root of each element.

#### Sub

```go
func Sub(a, b Tensor) Tensor
```

Sub returns a - b, element by element.

#### Sum

```go
func Sum(a Tensor, d int) Tensor
```

Sum adds up a along dimension d (-1 is the last). The result keeps the
dimension, with size 1: the sum of the rows of an m × n tensor, Sum(a, 0),
has shape [1 n].

#### Tanh

```go
func Tanh(a Tensor) Tensor
```

Tanh returns the hyperbolic tangent of each element.

#### Transpose

```go
func Transpose(a Tensor, d0, d1 int) Tensor
```

Transpose swaps two dimensions of a: Transpose(a, 0, 1) of an m × n
tensor is the n × m one.

#### Zeros

```go
func Zeros(shape []int) Tensor
```

Zeros makes a tensor of zeros.

#### (Tensor) Floats

```go
func (t Tensor) Floats() []float32
```

Floats returns the tensor's numbers, the last dimension running fastest.
On the card this reads them back (from the first chip: every chip holds
the same tensor).

#### (Tensor) Free

```go
func (t Tensor) Free()
```

Free gives the tensor's memory back. Tensors that are not freed stay
until the program ends.

#### (Tensor) Shape

```go
func (t Tensor) Shape() []int
```

Shape returns the tensor's dimensions.

#### (Tensor) TTNN

```go
func (t Tensor) TTNN() (shape []int, v []float32, handle int, onCard bool)
```

TTNN implements tt.NNTensor, for tt.LayerOf and tt.MatrixOf.

