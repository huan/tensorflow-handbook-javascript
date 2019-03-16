import { tf } from './config'

import { Vocabulary } from './vocabulary'

import {
  START_TOKEN,
  END_TOKEN,
}                 from './config'

type Seq2seqData = {
  input: string,
  output: string,
}

export function getCsvDataset (
  filename: string,
): tf.data.Dataset<Seq2seqData> {
  // We use "tab" as the "start sequence" character for the targets, and "\n"
  // as "end sequence" character.
  let dataset = tf.data.csv(filename, {
    hasHeader: false,
    columnNames: ['input', 'output'],
    delimiter: '\t',
  }) as any as tf.data.Dataset<Seq2seqData>

  return dataset
}


async function vocableDataset(
  dataset: tf.data.Dataset<Seq2seqData>,
) {
  const inputVoc = new Vocabulary()
  const outputVoc = new Vocabulary()

  // add START / STOP token
  outputVoc.fitToken(START_TOKEN)
  outputVoc.fitToken(END_TOKEN)

  let size = 0
  await dataset.forEachAsync(value => {
    inputVoc.fitText(value.input)
    outputVoc.fitText(value.output)
    size++
  })

  outputVoc.maxSeqLength += 2

  return {
    inputVoc,
    outputVoc,
    size,
  }
}


async function getSeq2seqDataset (
  dataset: tf.data.Dataset<Seq2seqData>,
  inputVoc: Vocabulary,
  outputVoc: Vocabulary,
) {
  const seq2seqDataset = dataset
  .map(value => {
    const inputSeq = inputVoc.sequenize(value.input, -1)
    const inputTensor = tf.tensor(inputSeq)

    const {
      decoderInput,
      decoderTarget,
    } = vectorizeForDecoder(value.output, outputVoc)

    const xs = {
      seq2seqInputs: inputTensor,
      seq2seqDecoderInputs: decoderInput,
    }
    const ys = decoderTarget

    return {xs, ys}
  })

  return seq2seqDataset
}

function vectorizeForDecoder (
  text: string,
  voc: Vocabulary,
) {
  const inputBuf = tf.buffer<tf.Rank.R1>([
    voc.maxSeqLength,
  ])
  const targetBuf = tf.buffer<tf.Rank.R2>([
    voc.maxSeqLength,
    voc.size,
  ])

  const indiceList = [
    voc.indice(START_TOKEN),
    ...voc.sequenize(text),
    voc.indice(END_TOKEN),
  ]

  for (const [t, indice] of indiceList.entries()) {
    inputBuf.set(indice, t)

    // shift left for target: not including START_OF_SEQ
    if (t > 0) {
      targetBuf.set(1, t - 1, indice)
    }
  }

  const decoderInput = inputBuf.toTensor()
  const decoderTarget = targetBuf.toTensor()

  return {
    decoderInput,
    decoderTarget,
  }
}


export async function getDataset (
  filename: string,
  limit = -1,
) {
  let dataset = getCsvDataset(
    filename,
  )

  if (limit > 0) {
    dataset = dataset.take(limit)
  }

  const {
    inputVoc,
    outputVoc,
    size,
  } = await vocableDataset(dataset)

  const seq2seqDataset = await getSeq2seqDataset(
    dataset,
    inputVoc,
    outputVoc,
  )

  return {
    dataset,
    seq2seqDataset,
    inputVoc,
    outputVoc,
    size,
  }
}



