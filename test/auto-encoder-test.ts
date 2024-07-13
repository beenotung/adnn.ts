import { createAutoEncoder } from 'auto-encoder.ts'
import { Network, nn } from '../adnn'
import { existsSync, writeFileSync } from 'fs'
import assert from 'assert'

let X = [
  [0, 0],
  [1, 1],
]

let autoEncoder = createAutoEncoder({
  scale: false,
  encoder: [
    { nOut: X[0].length, activation: 'tanh' },
    { nOut: X[0].length, activation: 'tanh' },
  ],
  decoder: [{ nOut: X[0].length, activation: 'tanh' }, { nOut: X[0].length }],
})

function test() {
  net.setTraining(false)
  let Y = autoEncoder.predict(X)

  let mse = 0
  for (let i = 0; i < Y.length; i++) {
    let x = X[i]
    let y = Y[i]
    for (let j = 0; j < x.length; j++) {
      let e = x[j] - y[j]
      mse += e * e
    }
  }
  mse /= Y.length

  return mse
}

let encoder = (autoEncoder as any).encoder as Network
let decoder = (autoEncoder as any).decoder as Network
let net = (autoEncoder as any).net as Network

let file = './net.json'
if (existsSync(file)) {
  let json = require(file)
  encoder = Network.deserializeJSON(json.encoder)
  decoder = Network.deserializeJSON(json.decoder)
  net = nn.sequence([encoder, decoder])

  assert.deepStrictEqual(
    encoder.serializeJSON(),
    json.encoder,
    'failed to load encoder',
  )
  Object.assign(autoEncoder, { encoder, decoder, net })
}

console.log('begin')
console.log(test())
// reloadNetwork()
debugger
autoEncoder.fit(X, {
  batchSize: 100,
  iterations: 5000,
  method: 'adagrad', // (default 'adagrad')
  stepSize: 0.01,
})
console.log('after')
console.log(test())

writeFileSync(
  file,
  JSON.stringify(
    {
      encoder: encoder.serializeJSON(),
      decoder: decoder.serializeJSON(),
    },
    null,
    2,
  ),
)
