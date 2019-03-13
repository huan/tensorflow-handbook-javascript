
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

import { ArgumentParser } from 'argparse'

import {
  tf,
  START_OF_SEQ,
  END_OF_SEQ,
}                      from './config'

import { createModel }  from './model/'

import {
  getDataset,
  vectorizeInput,
}                     from './data'

import { Vocabulary } from './vocabulary'

let FLAGS = {} as any

async function seq2seq (
  input: string,
  encoderModel: tf.Model,
  decoderModel: tf.Model,
  inputVoc: Vocabulary,
  outputVoc: Vocabulary,
): Promise<string> {
  const inputSeq = vectorizeInput(input, inputVoc)

  const batchedInputSeq = inputSeq.expandDims(0)
  let state = encoderModel.predict(batchedInputSeq) as tf.Tensor<tf.Rank.R2>

  // let state = encoderModel.predictOnBatch(inputSeq) as tf.Tensor<tf.Rank.R2>

  let decoderInput = outputVoc.indice(START_OF_SEQ)

  let decodedTensor: tf.Tensor<tf.Rank.R3>
  let decodedToken: string
  let decodedSentence = ''

  do {
    [decodedTensor, state] = decoderModel.predict([
      tf.tensor([decoderInput]),
      state,
    ]) as [
      tf.Tensor<tf.Rank.R3>,
      tf.Tensor<tf.Rank.R2>,
    ]

    let decodedIndice = await decodedTensor
                                .squeeze()
                                .argMax()
                                .array() as number

    if (decodedIndice === 0) {
      // 0 for padding, should be treated as END
      decodedToken = END_OF_SEQ
    } else {
      decodedToken = outputVoc.token(decodedIndice)
    }

    decodedSentence += decodedToken

    // save decoded data for next time step
    decoderInput = decodedIndice
  } while (
       decodedToken !== END_OF_SEQ
    && decodedSentence.length < outputVoc.size
  )

  return decodedSentence
}

async function main () {
  const {
    dataset,
    seq2seqDataset,
    size,
    inputVoc,
    outputVoc,
  } = await getDataset (
    'file://' + FLAGS.data_path,
    FLAGS.num_samples,
  )

  const {
    seq2seqModel,
    encoderModel,
    decoderModel,
  } = createModel (
    inputVoc,
    outputVoc,
    FLAGS.latent_dim,
  )

  seq2seqModel.summary()

  console.log('Sample num:', size)

  // Run training.
  seq2seqModel.compile({
    optimizer: 'rmsprop',
    loss: 'categoricalCrossentropy',
    // loss: 'sparseCategoricalCrossentropy',
  })

  await seq2seqDataset
  .batch(FLAGS.batch_size)
  .take(1)
  .forEachAsync((input: any) => {
    // console.log(input[0])
    console.log('seq2seq dataset sample:')
    console.log('seq2seqInputs', input[0].seq2seqInputs.shape)
    console.log('seq2seqDecoderInputs', input[0].seq2seqDecoderInputs.shape)

    // console.log(input[1])
    // console.log('ys', input[1].shape)
  })

  if (FLAGS.epochs > 0) {
    await seq2seqModel.fitDataset(
      seq2seqDataset
      .batch(FLAGS.batch_size)
      .prefetch(FLAGS.batch_size * 2),
      {
        epochs: FLAGS.epochs,
        // validationSplit: 0.2,
      },
    )
  }

  // FIXME: Layer decoderLstm was passed non-serializable keyword arguments: [object Object].
  // FIXME: They will not be included in the serialized model (and thus will be missing at deserialization time).

  // Huan: be aware that the Node need a `file://` prefix to local filename
  await seq2seqModel.save('file://' + FLAGS.artifacts_dir)

  const csvList = await dataset.take(FLAGS.num_test_sentences).toArray()

  for (const csvData of csvList) {
    const input = csvData.input
    const output = csvData.output

    const decodedOutput = await seq2seq(
      input,
      encoderModel,
      decoderModel,
      inputVoc,
      outputVoc,
    )

    console.log('-')
    console.log('Input sentence:', input)
    console.log('Target sentence:', output)
    console.log('Decoded sentence:', decodedOutput)
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
