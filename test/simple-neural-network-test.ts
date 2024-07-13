import { Tensor, TrainingData, nn, opt } from '../adnn'

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
