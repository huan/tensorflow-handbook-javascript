import { tf } from '../config'

import { Vocabulary } from '../vocabulary'

export function createLayers (
  inputVoc: Vocabulary,
  outputVoc: Vocabulary,
  latentDim: number,
) {
  /**
   * Encoder Layers
   */
  const encoderEmbeddingLayer = tf.layers.embedding({
    inputDim: inputVoc.size,
    outputDim: latentDim,
    name: 'encoderEmbedding',
  })

  const encoderRnnLayer = tf.layers.gru({
    units: latentDim,
    returnState: true,
    // recurrentInitializer: 'glorotUniform',
    name: 'encoder',
  })

  /**
   * Decoder Layers
   */
  const decoderEmbeddingLayer = tf.layers.embedding({
    inputDim: outputVoc.size,
    outputDim: latentDim,
    name: 'decoderEmbedding',
  })

  const decoderRnnLayer = tf.layers.gru({
    units: latentDim,
    returnSequences: true,
    returnState: true,
    // recurrentInitializer: 'glorotUniform',
    name: 'decoder',
  })

  const decoderDenseLayer = tf.layers.dense({
      units: outputVoc.size,
      activation: 'softmax',
      name: 'decoderDense',
  })

  return {
    encoderEmbeddingLayer,
    encoderRnnLayer,

    decoderEmbeddingLayer,
    decoderRnnLayer,
    decoderDenseLayer,
  }
}
