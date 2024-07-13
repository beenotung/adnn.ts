import { readFileSync, writeFileSync } from 'fs'

let Tensor = require('adnn/tensor')

let file = require.resolve('adnn/tensor')
let code = readFileSync(file).toString()
let lines = code.split('\n')

let body = Object.entries(Tensor.prototype)
  .map(([name, value]) => {
    let fn = value as Function
    let args = fn
      .toString()
      .split(')')[0]
      .split('(')
      .pop()!
      .split(',')
      .map(s => s.trim())
      .filter(s => s)
      .map(name => {
        let type =
          {
            dims: 'number[]',
            other: 'Node',
            offset: 'number',
            val: 'Value',
            x: 'Value',
            t: 'Tensor',
            coords: 'number[]',
            arr: 'unknown[]',
          }[name] || 'unknown_'
        return `${name}: ${type}`
      })
    let result =
      fn
        .toString()
        .split('return')[1]
        ?.replace(';', ' ')
        .replace('(', ' ')
        .split(' ')
        .map(s => s.trim())
        .filter(s => s) || []
    if (result[0] == 'new') {
      result.splice(0, 1)
    }
    let resultType = 'unknown_'
    if (resultType === 'unknown_') {
      if (
        lines.some(
          line =>
            line.includes(`addUnaryMethod('${name}'`) ||
            line.includes(`addUnaryMethod('${name.replace(/eq$/, '')}'`),
        )
      ) {
        resultType = 'Tensor'
      } else if (
        lines.some(
          line =>
            line.includes(`addBinaryMethod('${name}'`) ||
            line.includes(`addBinaryMethod('${name.replace(/eq$/, '')}'`),
        )
      ) {
        resultType = 'Tensor'
      } else if (
        lines.some(line =>
          line.includes(`addReduction('${name.replace(/reduce$/, '')}'`),
        )
      ) {
        resultType = 'number'
      }
    }
    if (resultType == 'unknown_') {
      resultType =
        {
          fill: 'this',
          zero: 'this',
          toString: 'string',
          copy: 'this',
          refCopy: 'this',
          clone: 'this',
          refClone: 'this',
          toArray: 'Value[]',
          toFlatArray: 'Value[]',
          softmax: 'Tensor',
          transpose: 'Tensor',
          diagonal: 'Tensor',
          inverse: 'Tensor',
          determinant: 'number',
          dot: 'Tensor',
          cholesky: 'Tensor',
          tril: 'Tensor',
          get: 'Value',
          set: 'void',
        }[name] ||
        result[0] ||
        'unknown_'
    }
    if (resultType.includes('.')) {
      resultType = 'unknown_'
      console.log('?', name, result)
    }
    return `  ${name}(${args}): ${resultType}`
  })
  .join('\n')

code =
  `
import { Value } from '../neural-network'

export interface Tensor {
  /* from constructor */
  dims: number[]
  length: number
  data: Float64Array
  /* from prototype */
  get rank(): number
${body}
}
`.trim() + '\n'

writeFileSync('dev/code-tensor.ts', code)
