import { Vocabulary } from '../vocabulary'

import { createLayers } from './layers'
import { getSeq2seqModel } from './seq2seq-model'
import { getEncoderModel } from './encoder-model'
import { getDecoderModel } from './decoder-model'
export { seq2seqDecoder } from './seq2seq-decoder'

export function createModel (
  inputVoc: Vocabulary,
  outputVoc: Vocabulary,
  latentDim: number,
) {
  const {
    encoderEmbeddingLayer,
    encoderRnnLayer,
    decoderDenseLayer,
    decoderEmbeddingLayer,
    decoderRnnLayer,
  } = createLayers(
    inputVoc,
    outputVoc,
    latentDim,
  )

  const seq2seqModel = getSeq2seqModel({
    encoderEmbeddingLayer,
    encoderRnnLayer,
    decoderDenseLayer,
    decoderEmbeddingLayer,
    decoderRnnLayer,
    inputVoc,
    outputVoc,
  })

  const encoderModel = getEncoderModel({
    encoderEmbeddingLayer,
    encoderRnnLayer,
    inputVoc,
  })

  const decoderModel = getDecoderModel({
    decoderDenseLayer,
    decoderEmbeddingLayer,
    decoderRnnLayer,
    latentDim,
  })

  return {
    seq2seqModel,
    encoderModel,
    decoderModel,
  }
}

