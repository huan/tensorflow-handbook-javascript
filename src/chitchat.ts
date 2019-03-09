
/**
 * Train a simple LSTM model for character-level language translation.
 * This is based on the Tensorflow.js example at:
 *   https://github.com/tensorflow/tfjs-examples/blob/master/translation/python/translation.py
 *
 * The training data can be downloaded with a command like the following example:
 *   wget http://www.manythings.org/anki/fra-eng.zip
 *
 * Author: Huan LI <zixia@zixia.net>
 * 2018, https://github.com/huan
 *
 */


import fs     from 'fs'
import path   from 'path'

import { ArgumentParser } from 'argparse'
import readline           from 'readline'

const { zip } = require('zip-array')
const invertKv = require('invert-kv')

import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-node'

let FLAGS = {} as any

async function readData (
  dataFile: string,
) {
  // Vectorize the data.
  const input_texts: string[] = []
  const target_texts: string[] = []

  const input_characters = new Set<string>()
  const target_characters = new Set<string>()

  const fileStream = fs.createReadStream(dataFile)
  const rl = readline.createInterface({
    input:    fileStream,
    output:   process.stdout,
    terminal: false,
  })

  let lineNumber = 0
  rl.on('line', line => {
    if (++lineNumber > FLAGS.num_samples) {
      rl.close()
      return
    }

    let [input_text, target_text] = line.split('\t')
    // We use "tab" as the "start sequence" character for the targets, and "\n"
    // as "end sequence" character.
    target_text = target_text + '\n'

    input_texts.push(input_text)
    target_texts.push(target_text)

    for (const char of input_text) {
      if (!input_characters.has(char)) {
        input_characters.add(char)
      }
    }
    for (const char of target_text) {
      if (!target_characters.has(char)) {
        target_characters.add(char)
      }
    }
  })

  await new Promise(r => rl.on('close', r))

  const input_character_list = [...input_characters].sort()
  const target_character_list = [...target_characters].sort()

  const num_encoder_tokens = input_character_list.length
  const num_decoder_tokens = target_character_list.length

  // Math.max() does not work with very large arrays because of the stack limitation
  const max_encoder_seq_length = input_texts.map(text => text.length)
                                            .reduceRight((prev, curr) => curr > prev ? curr : prev, 0)
  const max_decoder_seq_length = target_texts.map(text => text.length)
                                              .reduceRight((prev, curr) => curr > prev ? curr : prev, 0)

  console.log('Number of samples:', input_texts.length)
  console.log('Number of unique input tokens:', num_encoder_tokens)
  console.log('Number of unique output tokens:', num_decoder_tokens)
  console.log('Max sequence length for inputs:', max_encoder_seq_length)
  console.log('Max sequence length for outputs:', max_decoder_seq_length)

  const input_token_index = input_character_list.reduceRight(
    (prev, curr, idx) => (prev[curr] = idx + 1, prev),
    {} as {[char: string]: number},
  )
  const target_token_index = target_character_list.reduceRight(
    (prev, curr, idx) => (prev[curr] = idx + 1, prev),
    {} as {[char: string]: number},
  )

  // Save the token indices to file.
  const metadata_json_path = path.join(
    FLAGS.artifacts_dir,
    'metadata.json',
  )

  if (!fs.existsSync(path.dirname(metadata_json_path))) {
    fs.mkdirSync(path.dirname(metadata_json_path))
  }

  const metadata = {
    'input_token_index': input_token_index,
    'target_token_index': target_token_index,
    'max_encoder_seq_length': max_encoder_seq_length,
    'max_decoder_seq_length': max_decoder_seq_length,
  }

  fs.writeFileSync(metadata_json_path, JSON.stringify(metadata))
  console.log('Saved metadata at: ', metadata_json_path)

  const encoder_input_data_buf = tf.buffer<tf.Rank.R2>([
    input_texts.length,
    max_encoder_seq_length,
  ])
  const decoder_target_data_buf = tf.buffer<tf.Rank.R3>([
    input_texts.length,
    max_decoder_seq_length,
    num_decoder_tokens,
  ])

  for (
    const [i, [input_text, target_text]]
    of (zip(input_texts, target_texts).entries() as IterableIterator<[number, [string, string]]>)
  ) {
    for (const [t, char] of input_text.split('').entries()) {
      // encoder_input_data[i, t, input_token_index[char]] = 1.
      encoder_input_data_buf.set(input_token_index[char], i, t)
    }

    for (const [t, char] of target_text.split('').entries()) {
      // decoder_target_data is ahead of decoder_input_data by one timestep
      decoder_target_data_buf.set(1, i, t, target_token_index[char])
    }

  }

  const encoder_input_data = encoder_input_data_buf.toTensor()
  const decoder_target_data = decoder_target_data_buf.toTensor()

  return {
    input_texts,
    max_encoder_seq_length,
    max_decoder_seq_length,
    num_encoder_tokens,
    num_decoder_tokens,
    input_token_index,
    target_token_index,
    encoder_input_data,
    decoder_target_data,
  }
}

function seq2seqModel (
  num_encoder_tokens: number,
  num_decoder_tokens: number,
  max_decoder_seq_length: number,
) {
  const model = tf.sequential()
  model.add(tf.layers.embedding({
    inputDim: num_encoder_tokens + 1,
    outputDim: 64,
    name: 'embedding',
    // maskZero: true,
  }))
  model.add(tf.layers.lstm({
    units: 64,
    name: 'encoder',
    // goBackwards: true,
  }))
  model.add(tf.layers.repeatVector({
    n: max_decoder_seq_length,
  }))
  model.add(tf.layers.lstm({
    units: 64,
    returnSequences: true,
    name: 'decoder',
  }))
  model.add(tf.layers.dense({
    units: num_decoder_tokens,
    activation: 'softmax',
  }))

  return model
}

async function decode_sequence (
  inputs: tf.Tensor,
  model: tf.Model,
  reverse_target_char_index: {[indice: number]: string},
) {
  const outputs = model.predict(inputs) as tf.Tensor<tf.Rank.R3>
  // console.log(outputs.shape)
  const indiceList = await outputs.argMax(-1).squeeze().array() as number[]

  const decoded_sequence = indiceList.map(indice => reverse_target_char_index[indice])

  // console.log('EOS indice', 1)
  // console.log('indiceList', indiceList)
  // console.log('charList', decoded_sequence)
  const len = decoded_sequence.indexOf('\n')
  const decoded_sentence = decoded_sequence.slice(0, len).join('')

  return decoded_sentence
}

// function loss (
//   yTrue: tf.Tensor,
//   yPred: tf.Tensor,
// ): tf.Tensor {
//   let lossVal = tf.metrics.categoricalCrossentropy(yTrue, yPred)
//   const num = yTrue.sum(-1).sum()
//   return lossVal.sum().div(num)
// }

async function main () {
  const {
    input_texts,
    max_decoder_seq_length,
    num_encoder_tokens,
    num_decoder_tokens,
    target_token_index,
    encoder_input_data,
    // decoder_input_data,
    decoder_target_data,
  } = await readData(FLAGS.data_path)

  const model = seq2seqModel(
    num_encoder_tokens,
    num_decoder_tokens,
    max_decoder_seq_length,
  )

  model.summary()

  // Run training.
  model.compile({
    optimizer: 'rmsprop',
    loss: 'categoricalCrossentropy',
    // loss: 'sparseCategoricalCrossentropy',
  })

  await model.fit(
    encoder_input_data,
    decoder_target_data,
    {
      batchSize: FLAGS.batch_size,
      epochs: FLAGS.epochs,
      validationSplit: 0.2,
    },
  )

  // FIXME: Layer decoderLstm was passed non-serializable keyword arguments: [object Object].
  // FIXME: They will not be included in the serialized model (and thus will be missing at deserialization time).

  // Huan: be aware that the Node need a `file://` prefix to local filename
  await model.save('file://' + FLAGS.artifacts_dir)

  // Reverse-lookup token index to decode sequences back to
  // something readable.
  const reverse_target_char_index = invertKv(target_token_index) as {[indice: number]: string}

  for (let seq_index = 0; seq_index < FLAGS.num_test_sentences; seq_index++) {
    // Take one sequence (part of the training set)
    // for trying out decoding.
    const input_seq = encoder_input_data.slice(seq_index, 1)

    // Get expected output
    const target_seq_voc = decoder_target_data.slice(seq_index, 1).squeeze([0]) as tf.Tensor2D
    const target_seq_tensor = target_seq_voc.argMax(-1) as tf.Tensor1D

    const target_seq_list = await target_seq_tensor.array()

    // One-hot to index
    const target_seq = target_seq_list.map(indice => reverse_target_char_index[indice])

    // Array to string
    const target_seq_str = target_seq.join('').replace(/\n.*$/, '')
    const decoded_sentence = await decode_sequence(
      input_seq,
      model,
      reverse_target_char_index,
    )
    console.log('-')
    console.log('Input sentence:', input_texts[seq_index])
    console.log('Target sentence:', target_seq_str)
    console.log('Decoded sentence:', decoded_sentence)
  }
}


const parser = new ArgumentParser({
  version: '0.0.1',
  addHelp: true,
  description: 'Keras seq2seq translation model training and serialization',
})

parser.addArgument(
  ['data_path'],
  {
    type: 'string',
    help: 'Path to the training data, e.g., ~/ml-data/fra-eng/fra.txt',
  },
)
parser.addArgument(
  '--batch_size',
  {
    type: 'int',
    defaultValue: 64,
    help: 'Training batch size.'
  }
)
parser.addArgument(
  '--epochs',
  {
    type: 'int',
    defaultValue: 20,
    help: 'Number of training epochs.',
  },
)
parser.addArgument(
  '--latent_dim',
  {
    type: 'int',
    defaultValue: 256,
    help: 'Latent dimensionality of the encoding space.',
  },
)
parser.addArgument(
  '--num_samples',
  {
    type: 'int',
    defaultValue: 10000,
    help: 'Number of samples to train on.',
  }
)
parser.addArgument(
  '--num_test_sentences',
  {
    type: 'int',
    defaultValue: 100,
    help: 'Number of example sentences to test at the end of the training.',
  },
)
parser.addArgument(
  '--artifacts_dir',
  {
    type: 'string',
    defaultValue: '/tmp/translation.keras',
    help: 'Local path for saving the TensorFlow.js artifacts.',
  },
)

;[FLAGS,] = parser.parseKnownArgs()
main()
