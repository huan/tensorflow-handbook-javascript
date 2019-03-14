
/**
 * Concist ChitChat in JavaScript/TypeScript
 *
 * Author: Huan LI <zixia@zixia.net>
 * 2019, https://github.com/huan
 *
 */

import { ArgumentParser } from 'argparse'

import {
  createModel,
  seq2seqDecoder,
  getDataset,
}                   from '../'

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

  // Run training.
  seq2seqModel.compile({
    optimizer: 'rmsprop',
    loss: 'categoricalCrossentropy',
    // loss: 'sparseCategoricalCrossentropy',
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
      },
    )
  }

  // FIXME: Layer decoderLstm was passed non-serializable keyword arguments: [object Object].
  // FIXME: They will not be included in the serialized model (and thus will be missing at deserialization time).

  // Huan: be aware that the Node need a `file://` prefix to local filename
  await seq2seqModel.save('file://' + args.artifacts_dir)
  console.log('Model saved to ', args.artifacts_dir)

  const csvList = await dataset.take(args.num_test_sentences).toArray()

  for (const csvData of csvList) {
    const input = csvData.input
    const output = csvData.output

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

  return 0
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
