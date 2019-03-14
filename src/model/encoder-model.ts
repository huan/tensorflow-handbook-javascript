import { tf } from '../config'

import { Vocabulary } from '../vocabulary'

interface EncoderModelOptions {
  encoderEmbeddingLayer: tf.layers.Layer,
  encoderRnnLayer: tf.layers.Layer,
  inputVoc: Vocabulary,
}

/**
 * Encoder Model
 */
export function getEncoderModel (options: EncoderModelOptions): tf.LayersModel {
  const {
    encoderEmbeddingLayer,
    encoderRnnLayer,
    inputVoc,
  } = options

  const encoderInputs = tf.layers.input({
    shape: [inputVoc.maxSeqLength],
    name: 'encoderInputs',
  })
  const encoderEmbedding = encoderEmbeddingLayer.apply(encoderInputs)
  const [, encoderState] = encoderRnnLayer.apply(encoderEmbedding) as tf.SymbolicTensor[]

  const encoderModel = tf.model({
    inputs: encoderInputs,
    outputs: encoderState,
  })

  return encoderModel
}
