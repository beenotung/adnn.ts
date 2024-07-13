import { readFileSync, writeFileSync } from 'fs'

let ad = require('adnn/ad')

let functions_code = readFileSync(require.resolve('adnn/ad/functions'))
let functions_lines = functions_code.toString().split('\n')

let code = `
import { Value, Tensor, Scalar } from '../neural-network'
`

function scanObject(name: string) {
  let ResultType = (name[0].toUpperCase() + name.slice(1)) as
    | 'Tensor'
    | 'Scalar'
  let Input1Type = ResultType == 'Tensor' ? 'Tensor' : 'number'
  let Input2Type = ResultType == 'Tensor' ? 'number | Tensor' : 'number'
  let object = ad[name]
  let body = Object.entries(object)
    .flatMap(([name, value]) => {
      let unops = ['neg']
      let binops = ['add', 'sub', 'mul', 'div']
      var unaryFns = [
        'floor',
        'ceil',
        'round',
        'sqrt',
        'exp',
        'log',
        'abs',
        'sin',
        'cos',
        'tan',
        'asin',
        'acos',
        'atan',
        'sinh',
        'cosh',
        'tanh',
        'asinh',
        'acosh',
        'atanh',
        'sigmoid',
      ]
      var binaryFns = ['pow', 'min', 'max', 'atan2']

      if ([...unops, ...unaryFns].includes(name)) {
        if (ResultType == 'Scalar') {
          return (
            '  ' +
            `
  ${name}(x: number): number
  ${name}(x: Node): Node
`.trim()
          )
        }
        return `  ${name}(x: ${Input1Type}): ${ResultType}`
      }
      if ([...binops, ...binaryFns].includes(name)) {
        if (ResultType == 'Scalar') {
          return (
            '  ' +
            `
  ${name}(x: number, y: number): number
  ${name}(x: number | Node, y: number | Node): Node
`.trim()
          )
        }
        return `  ${name}(x: ${Input1Type}, y: ${Input2Type}): ${ResultType}`
      }

      if (
        ResultType == 'Scalar' &&
        functions_code.includes(`fns.scalar.${name} = func.liftBinaryFunction(`)
      ) {
        return `  ${name}(x: number, y: number): boolean`
      }

      if (ResultType == 'Scalar' && name in Math) {
        return `  ${name}: Math['${name}']`
      }

      if (
        ResultType == 'Tensor' &&
        functions_code.includes(`fns.tensor.${name} = func.newUnaryFunction(`)
      ) {
        return [
          /* overload */
          `${name}(x: Node): Node`,
          `${name}(x: Value): Value`,
        ]
      }

      if (
        ResultType == 'Tensor' &&
        functions_code.includes(`fns.tensor.${name} = func.liftUnaryFunction(`)
      ) {
        return `${name}(x: Node | Value): number`
      }

      let knownTypes: {
        [ResultType: string]: { [name: string]: string[] | string }
      } = {
        Scalar: {
          sum: [
            `(args: Array<number | Node>): number`,
            `(...args: Array<number | Node>): number`,
          ],
          isNaN: '(x: Value): x is number',
          isFinite: '(x: Value): x is number',
        },
        Tensor: {
          isNaN: '(x: Tensor): Tensor',
          isFinite: '(x: Tensor): Tensor',
          toScalars: '(t: Tensor): number[]',
          fromScalars: [
            '(args: Array<number>): Tensor',
            '(...args: Array<number>): Tensor',
          ],
          concat: [
            '(args: Array<Tensor | Node>): Tensor',
            '(...args: Array<Tensor | Node>): Tensor',
          ],
          range:
            '(t: Array<Tensor | Node>, start: number, end: number): Tensor',
          split: '(t: Array<Tensor | Node>, lengths: number[]): Tensor[]',
          get: '(t: Array<Tensor | Node>, index: number): Tensor[]',
          dot: '(x: Tensor, y: Tensor): Tensor',
          reshape: '(t: Tensor, dims: number[]): Tensor',
        },
      }

      let knownType = knownTypes[ResultType][name]
      if (knownType) {
        return (Array.isArray(knownType) ? knownType : [knownType])
          .map(type => `  ${name}${type}`)
          .join('\n')
      }

      return `  ${name} ?`
    })
    .map(line => '  ' + line.trim())
    .join('\n')
  code += `
export type ${ResultType}Functions = {
${body}
}
`
}

scanObject('scalar')
scanObject('tensor')

writeFileSync('dev/code-ad.ts', code)
