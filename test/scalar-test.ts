import { ScalarNode, ad, scalar } from '../adnn'

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
