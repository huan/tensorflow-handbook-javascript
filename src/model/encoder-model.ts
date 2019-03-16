import { tf } from '../config'

interface EncoderModelOptions {
  encoderEmbeddingLayer: tf.layers.Layer,
  encoderRnnLayer: tf.layers.Layer,
}

/**
 * Encoder Model
 */
export function getEncoderModel (options: EncoderModelOptions): tf.LayersModel {
  const {
    encoderEmbeddingLayer,
    encoderRnnLayer,
  } = options

  const encoderInputs = tf.layers.input({
    shape: [null] as any as number[],
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
