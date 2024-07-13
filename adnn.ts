export let Tensor: TensorConstructor = require('adnn/tensor')
export let ad: AD = require('adnn/ad')
export let nn: NN = require('adnn/nn')
export let opt: Opt = require('adnn/opt')
export let Network: NetworkConstructor = require('adnn/nn/network')

type AD = {
  lift(x: number): ScalarNode
  lift(x: Tensor): TensorNode
  value(x: number): number
  value(x: ScalarNode): ScalarNode['x']
  derivative(x: ScalarNode): ScalarNode['dx']
  derivative(x: TensorNode): TensorNode['dx']
  /** @description Create randomly-initialized params */
  params(dims: number[], name?: string): TensorNode
  scalar: {
    add: ScalarBinaryOp
    sub: ScalarBinaryOp
    mul: ScalarBinaryOp
    div: ScalarBinaryOp
    sqrt: ScalarUnaryOp
    sum: ScalarUnaryReduction
  }
  tensor: {
    add: TensorBinaryOp
    sub: TensorBinaryOp
    mul: TensorBinaryOp
    div: TensorBinaryOp
    sqrt: TensorUnaryOp
    sumreduce: TensorUnaryReduction
  }
}

interface ScalarUnaryReduction {
  (...xs: number[]): number
  (xs: number[]): number
  (...xs: scalar[]): ScalarNode
  (xs: scalar[]): ScalarNode
}

interface ScalarUnaryOp {
  (x: number): number
  (x: ScalarNode): ScalarNode
}

interface ScalarBinaryOp {
  (x: number, y: number): number
  (x: scalar, y: scalar): ScalarNode
}

interface TensorUnaryReduction {
  (x: Tensor): number
  (x: TensorNode): ScalarNode
}

interface TensorUnaryOp {
  (x: Tensor): Tensor
  (x: TensorNode): TensorNode
}

interface TensorBinaryOp {
  (x: Tensor, y: number | Tensor): Tensor
  (x: TensorNode, y: number | Tensor | TensorNode): TensorNode
}

type NN = {
  relu: Network
  tanh: Network
  sigmoid: Network
  /**
   * @description Sigmoid, shifted and scaled to the range (-1, 1)
   *  Same output range as tanh, but numerically stable
   * (i.e. doesn't give NaNs for large inputs).
   */
  sigmoidCentered: Network
  softmax: Network
  mlp(
    nIn: number,
    layerdefs: NetworkLayerDef[],
    name?: string,
    debug?: boolean,
  ): Network
  sequence(networks: Network[], name?: string, debug?: boolean): CompoundNetwork
  linear(nIn: number, nOut: number, name?: string): LinearNetwork
}

type Opt = {
  nnTrain(
    network: Network,
    trainingData: TrainingData,
    lossFn: LossFn,
    options: TrainOptions,
  ): void
  sgd(options?: {
    stepSize?: number
    stepSizeDecay?: number
    mu?: number
  }): OptimizeMethod
  adagrad(options?: { stepSize?: number }): OptimizeMethod
  rmsprop(options?: { stepSize?: number; decayRate?: number }): OptimizeMethod
  adam(options?: { stepSize?: number }): OptimizeMethod
  classificationLoss: LossFn
  regressionLoss: LossFn
}

export type OptimizationMethodName = 'sgd' | 'adagrad' | 'rmsprop' | 'adam'

export type TrainOptions = {
  iterations?: number
  batchSize?: number
  method?: OptimizeMethod
}

export type OptimizeMethod = {}

export type LossFn =
  | ((outputProbs: Tensor, trueClassIndex: number) => Tensor | number)
  | ((outputProbs: Tensor, trueOutput: number[]) => Tensor | number)

export type TrainingData = {
  input: NetworkInput
  output: NetworkOutput
}[]

export type NetworkLayerDef = {
  nOut: number
  activation?: Activation
}

export type Activation = NN[ActivationFunctionName]

export type ActivationFunctionName =
  | 'relu'
  | 'tanh'
  | 'sigmoid'
  | 'sigmoidCentered'
  | 'softmax'

export interface NetworkConstructor {
  new (...args: unknown[]): Network
  deserializeJSON(json: unknown): Network
}

export interface Network {
  name: string
  isTraining: boolean
  eval(input: NetworkInput): NetworkOutput
  setParameters(params: unknown): void
  getParameters(): unknown
  setTraining(isTraining: boolean): void
  serializeJSON(): unknown
}

export type NetworkInput = Tensor
export type NetworkOutput = Tensor | ClassIndex

/** @description starting from 0 */
export type ClassIndex = number

export interface CompoundNetwork extends Network {
  networks: Network[]
}

export interface LinearNetwork extends Network {
  inSize: number
  outSize: number
  weights: TensorNode
  biases: TensorNode
}

export interface TensorConstructor {
  new (dims: number[]): Tensor
}

export interface Tensor {
  dims: number[]
  length: number
  fromArray(arr: number[]): this
  fromFlatArray(arr: number[]): this
  fillRandom(): this
  toArray(): number[]
  toFlatArray(): number[]
  data: Float64Array
}

export interface TensorNode extends Node<Tensor> {
  x: Tensor
  dx: Tensor
}

export type scalar = ScalarNode | number

export interface ScalarNode extends Node<number> {
  x: number
  dx: number
  backprop(): void
}

export interface Node<T> {
  x: T
  parents?: Node<unknown>[]
  inputs?: unknown[]
  backward?: unknown
  outDegree: number
  name: string
}
