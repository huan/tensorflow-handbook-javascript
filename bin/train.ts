
/**
 * Concist ChitChat in JavaScript/TypeScript
 *
 * Author: Huan LI <zixia@zixia.net>
 * 2019, https://github.com/huan
 *
 */
import fs from 'fs'

import {
  Tensor,
}                     from '@tensorflow/tfjs'

import { ArgumentParser } from 'argparse'

import {
  createModel,
  seq2seqDecoder,
  getDataset,
}                   from '../'
import { tf } from '../src/config';
import { Loader } from '../src/loader';

function loss (
  yTrue: Tensor,
  yPred: Tensor,
): Tensor {
  const meanLoss = tf.metrics.categoricalCrossentropy(yTrue, yPred).mean()

  const mask = yTrue.sum(-1)
  const placingRate = mask.sum().div(mask.size)

  const lossPerToken = meanLoss.sum().div(placingRate)

  return lossPerToken
}

interface Args {
  // TODO: define the specific args types
  [key: string]: any
}

async function main (args: Args) {
  if (args.gpu) {
    console.log('Using GPU')
    require('@tensorflow/tfjs-node-gpu')
  } else {
    console.log('Using CPU')
    require('@tensorflow/tfjs-node')
  }

  const {
    dataset,
    seq2seqDataset,
    size,
    inputVoc,
    outputVoc,
  } = await getDataset (
    'file://' + args.data_path,
    args.num_samples,
  )

  const {
    seq2seqModel,
    encoderModel,
    decoderModel,
  } = createModel (
    inputVoc,
    outputVoc,
    args.latent_dim,
  )

  seq2seqModel.summary()

  console.log('Sample num:', size)
  console.log('Input vocabulary size:', inputVoc.size)
  console.log('Input vocabulary maxSeqLength:', inputVoc.maxSeqLength)
  console.log('Input vocabulary:', [...inputVoc.tokenIndice].slice(0, 10))
  console.log('Output vocabulary size:', outputVoc.size)
  console.log('Output vocabulary maxSeqLength:', outputVoc.maxSeqLength)
  console.log('Output vocabulary:', [...outputVoc.tokenIndice].slice(0, 10))

  const optimizer = new tf.AdamOptimizer(1e-3, 0.9, 0.999, 1e-8)
  // const optimizer = new tf.RMSPropOptimizer(1e-2)

  // Run training.
  seq2seqModel.compile({
    optimizer: optimizer,
    // loss: 'categoricalCrossentropy',
    loss,
  })

  await seq2seqDataset
  .batch(args.batch_size)
  .take(1)
  .forEachAsync((input: any) => {
    console.log(input)
    // console.log('seq2seq dataset sample:')
    // console.log('seq2seqInputs', input[0].seq2seqInputs.shape)
    // console.log('seq2seqDecoderInputs', input[0].seq2seqDecoderInputs.shape)

    // console.log(input[1])
    // console.log('ys', input[1].shape)
  })

  if (args.epochs > 0) {
    await seq2seqModel.fitDataset(
      seq2seqDataset
      .batch(args.batch_size)
      .prefetch(args.batch_size * 2),
      {
        epochs: args.epochs,
        // validationSplit: 0.2,
        // verbose: 1,
        callbacks: {
          onEpochEnd,
        }
      },
    )
  }

  // FIXME: Layer decoderLstm was passed non-serializable keyword arguments: [object Object].
  // FIXME: They will not be included in the serialized model (and thus will be missing at deserialization time).
  if (!fs.existsSync(args.artifacts_dir))  {
    fs.mkdirSync(args.artifacts_dir)
  }

  await Loader.save({
    encoder: encoderModel,
    decoder: decoderModel,
    model: seq2seqModel,
    inputVoc,
    outputVoc,
  }, args.artifacts_dir)

  console.log('Model saved to ', args.artifacts_dir)

  await showTestSentences()

  return 0

  async function onEpochEnd(epoch: number, logs?: tf.Logs): Promise<void> {
    console.log('epoch ', epoch, 'logs', logs)
    await showTestSentences()
    return
  }

  async function showTestSentences() {
    const testSentenceList = await dataset.take(args.num_test_sentences).toArray()

    for (const testSentence of testSentenceList) {
      const input = testSentence.input
      const output = testSentence.output

      const decodedOutput = await seq2seqDecoder(
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

}

function parseArguments () {
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
      defaultValue: 10,
      help: 'Number of example sentences to test at the end of the training.',
    },
  )
  parser.addArgument(
    '--artifacts_dir',
    {
      type: 'string',
      defaultValue: '/tmp/chitchat',
      help: 'Local path for saving the TensorFlow.js artifacts.',
    },
  )
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'Use tfjs-node-gpu to train the model. Requires CUDA/CuDNN.'
  })

  return parser.parseArgs()
}

main(parseArguments())
.then(process.exit)
.catch(e => {
  console.error(e)
  process.exit(1)
})
