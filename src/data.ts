import { tf } from './config'

import { Vocabulary } from './vocabulary'

import {
  START_OF_SEQ,
  END_OF_SEQ,
}                 from './config'

type Seq2seqData = {
  input: string,
  output: string,
}

export function getCsvDataset (
  filename: string,
  limit = -1,
): tf.data.Dataset<Seq2seqData> {
  // We use "tab" as the "start sequence" character for the targets, and "\n"
  // as "end sequence" character.
  let csvDataset = tf.data.csv(filename, {
    hasHeader: false,
    columnNames: ['input', 'output'],
    delimiter: '\t',
  }) as any as tf.data.Dataset<Seq2seqData>

  if (limit > 0) {
    csvDataset = csvDataset.take(limit)
  }

  csvDataset = csvDataset.map(value => {
    return {
      input: value.input,
      output: START_OF_SEQ + value.output + END_OF_SEQ,
    }
  })

  return csvDataset
}


async function getSeq2seqDataset (
  dataset: tf.data.Dataset<Seq2seqData>,
  inputVoc: Vocabulary,
  outputVoc: Vocabulary,
) {
  const seq2seqDataset = dataset
  .map(value => {
    const input = vectorizeInput(value.input, inputVoc)

    const {
      decoderInput,
      decoderTarget,
    } = vectorizeForDecoder(value.output, outputVoc)

    return [
      {
        seq2seqInputs: input,
        seq2seqDecoderInputs: decoderInput,
      },
      decoderTarget,
    ]
  })

  return seq2seqDataset

  function vectorizeInput (
    text: string,
    voc: Vocabulary,
  ): tf.Tensor<tf.Rank.R1> {
    const tokenList = [...voc.tokenizer.tokenize(text)]
                      .map(token => voc.indice(token))

    const tokenLength = tokenList.length
    const needFill = voc.maxSeqLength > tokenLength

    // if longer than max length, cut it
    // if shorter than max length, expand it(and then fill 0)
    tokenList.length = voc.maxSeqLength
    if (needFill) {
      tokenList.fill(0, tokenLength)
    }
    return tf.tensor(tokenList)
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

    const tokenList = [...voc.tokenizer.tokenize(text)]
    for (const [t, token] of tokenList.entries()) {
      const indice = voc.indice(token)
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
}

async function vocableDataset(
  dataset: tf.data.Dataset<Seq2seqData>,
) {
  const inputVoc = new Vocabulary()
  const outputVoc = new Vocabulary()

  let size = 0
  await dataset.forEachAsync(value => {
    inputVoc.fitText(value.input)
    outputVoc.fitText(value.output)
    size++
  })

  return {
    inputVoc,
    outputVoc,
    size,
  }
}

export async function getDataset (
  filename: string,
  limit = -1,
) {
  let dataset = getCsvDataset(
    filename,
    limit,
  )

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



