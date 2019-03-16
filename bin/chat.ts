import '@tensorflow/tfjs-node'

import { ArgumentParser } from 'argparse'

import {
  Loader,
  seq2seqDecoder,
}                   from '../'

interface Args {
  model_path: string,
}

async function main (args: Args) {
  const {
    encoder,
    decoder,
    inputVoc,
    outputVoc
  } = await Loader.load(args.model_path)

  encoder.summary()
  decoder.summary()

  const input = 'how are you ?'

  const decodedOutput = await seq2seqDecoder(
    input,
    encoder,
    decoder,
    inputVoc,
    outputVoc,
  )

  console.log('-')
  console.log('Input sentence:', input)
  console.log('Decoded sentence:', decodedOutput)
  return 0
}


function parseArguments () {
  const parser = new ArgumentParser({
    version: '0.0.1',
    addHelp: true,
    description: 'Keras seq2seq translation model training and serialization',
  })

  parser.addArgument(
    ['model_path'],
    {
      type: 'string',
      help: 'Path to the saved model, e.g., ~/tfjs/chitchat',
    },
  )

  return parser.parseArgs()
}

main(parseArguments())
.then(process.exit)
.catch(e => {
  console.error(e)
  process.exit(1)
})
