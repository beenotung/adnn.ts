# adnn.ts

adnn.ts provides TypeSafe Javascript-native neural networks on top of general scalar/tensor reverse-mode automatic differentiation. You can use just the AD code, or the NN layer built on top of it. This architecture makes it easy to define big, complex numerical computations and compute derivatives w.r.t. their inputs/parameters. adnn also includes utilities for optimizing/training the parameters of such computations.

[![npm Package Version](https://img.shields.io/npm/v/adnn.ts)](https://www.npmjs.com/package/adnn.ts)
[![Minified Package Size](https://img.shields.io/bundlephobia/min/adnn.ts)](https://bundlephobia.com/package/adnn.ts)
[![Minified and Gzipped Package Size](https://img.shields.io/bundlephobia/minzip/adnn.ts)](https://bundlephobia.com/package/adnn.ts)

This is Typescript wrapper on top of [adnn](https://github.com/dritchie/adnn)

## Features

- Support reverse-mode automatic differentiation
- Static Type Checking and Completion with Typescript
- Isomorphic package: works in Node.js and browsers
- Javascript-native (without clumsome native dependencies, no node-gpy, no cmake, no python, no cuda)

## Installation

```bash
npm install adnn.ts
```

You can also install `adnn.ts` with [pnpm](https://pnpm.io/), [yarn](https://yarnpkg.com/), or [slnpm](https://github.com/beenotung/slnpm)

## Usage Example

### Scalar code

The simplest use case for adnn:

```typescript
import { ScalarNode, ad, scalar } from 'adnn.ts'

// Can use normal number or lifted ScalarNode
function dist(x1: number, y1: number, x2: number, y2: number): number
function dist(x1: scalar, y1: scalar, x2: scalar, y2: scalar): ScalarNode
function dist(x1: scalar, y1: scalar, x2: scalar, y2: scalar): scalar {
  var xdiff = ad.scalar.sub(x1, x2)
  var ydiff = ad.scalar.sub(y1, y2)
  return ad.scalar.sqrt(
    ad.scalar.add(ad.scalar.mul(xdiff, xdiff), ad.scalar.mul(ydiff, ydiff)),
  )
}

// number in, number out
var number_output = dist(0, 1, 1, 4)
console.log(number_output) // 3.162...

// Use 'lifted' inputs to track derivatives
var x1 = ad.lift(0)
var y1 = ad.lift(1)
var x2 = ad.lift(1)
var y2 = ad.lift(4)

// scalar in, scalar out
var scalar_output = dist(x1, y1, x2, y2)
console.log(ad.value(scalar_output)) // still 3.162...

scalar_output.backprop() // Compute derivatives of inputs
console.log(ad.derivative(x1)) // -0.316...
```

### Tensor code

adnn also supports computations involving tensors, or a mixture of scalars and tensors:

```javascript
import { Tensor, TensorNode, ad } from 'adnn.ts'

function dot(vec: TensorNode) {
  var sq = ad.tensor.mul(vec, vec)
  return ad.tensor.sumreduce(sq)
}

function dist(vec1: TensorNode, vec2: TensorNode) {
  return ad.scalar.sqrt(dot(ad.tensor.sub(vec1, vec2)))
}

var vec1 = ad.lift(new Tensor([3]).fromFlatArray([0, 1, 1]))
var vec2 = ad.lift(new Tensor([3]).fromFlatArray([2, 0, 3]))
var out = dist(vec1, vec2)
console.log(ad.value(out)) // 3
out.backprop()
console.log(ad.derivative(vec1).toFlatArray()) // [-0.66, 0.33, -0.66]
```

### Simple neural network

adnn makes it easy to define simple, feedforward neural networks. Here's a basic multilayer perceptron that takes a feature vector as input and outputs class probabilities:

```javascript
import { Tensor, TrainingData, nn, opt } from 'adnn.ts'

var nInputs = 20
var nHidden = 10
var nClasses = 5

// Definition using basic layers
var net = nn.sequence([
  nn.linear(nInputs, nHidden),
  nn.tanh,
  nn.linear(nHidden, nClasses),
  nn.softmax,
])

// Alternate definition using 'nn.mlp' utility
net = nn.sequence([
  nn.mlp(nInputs, [{ nOut: nHidden, activation: nn.tanh }, { nOut: nClasses }]),
  nn.softmax,
])

// Train the parameters of the network from some dataset
// 'loadData' is a stand-in for a user-provided function that
//    loads in an array of {input: , output: } objects
// Here, 'input' is a feature vector, and 'output' is a class label
var trainingData = loadData(100)
opt.nnTrain(net, trainingData, opt.classificationLoss, {
  batchSize: 10,
  iterations: 100,
  method: opt.adagrad(),
})

// Predict class probabilities for new, unseen features
var features = new Tensor([nInputs]).fillRandom()
var classProbs = net.eval(features)

console.log({ features, classProbs })

function loadData(sampleSize: number): TrainingData {
  return new Array(sampleSize).fill(0).map(() => ({
    input: new Tensor([nInputs]).fillRandom(),
    output: Math.floor(Math.random() * nClasses),
  }))
}
```

_Below sections are still working in progress, you can read the js version in the meanwhile._

### Convolutional neural network

[js version](https://github.com/dritchie/adnn/blob/master/README.md#convolutional-neural-network)

### Recurrent neural network

[js version](https://github.com/dritchie/adnn/blob/master/README.md#recurrent-neural-network)

### The `ad` module

The `ad` module has its own documentation [here](https://github.com/dritchie/adnn/blob/master/ad/README.md)

### The `nn` module

The `nn` module has its own documentation [here](https://github.com/dritchie/adnn/blob/master/nn/README.md)

### The `opt` module

The `opt` module has its own documentation [here](https://github.com/dritchie/adnn/blob/master/opt/README.md)

### Tensors

[js version](https://github.com/dritchie/adnn/blob/master/README.md#tensors)

## Typescript Signature

Details see [adnn.ts](./adnn.ts)

## License

This project is licensed with [BSD-2-Clause](./LICENSE)

This is free, libre, and open-source software. It comes down to four essential freedoms [[ref]](https://seirdy.one/2021/01/27/whatsapp-and-the-domestication-of-users.html#fnref:2):

- The freedom to run the program as you wish, for any purpose
- The freedom to study how the program works, and change it so it does your computing as you wish
- The freedom to redistribute copies so you can help others
- The freedom to distribute copies of your modified versions to others
