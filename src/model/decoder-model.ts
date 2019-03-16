import { tf } from '../config'

interface DecoderModelOptions {
  decoderEmbeddingLayer: tf.layers.Layer,
  decoderRnnLayer: tf.layers.Layer,
  decoderDenseLayer: tf.layers.Layer,
  latentDim: number,
}

/**
 * Decoder Model
 */
export function getDecoderModel (options: DecoderModelOptions): tf.LayersModel {
  const {
    decoderEmbeddingLayer,
    decoderRnnLayer,
    decoderDenseLayer,
    latentDim,
  } = options

  const decoderInput = tf.layers.input({
    shape: [null] as any as number[], // FIXME: see issue of tfjs Shape
    name: 'decoderInputs',
  })
  const decoderStateInput = tf.layers.input({
    shape: [latentDim],
    name: 'decoderState',
  }) as tf.SymbolicTensor

  const decoderEmbedding = decoderEmbeddingLayer.apply(decoderInput) as tf.SymbolicTensor

  const [decoderOutputs, decoderStateOutput] = decoderRnnLayer.apply(
    [decoderEmbedding, decoderStateInput],
    {
      returnState: true,
    },
  ) as tf.SymbolicTensor[]
  const decoderDenseOutputs = decoderDenseLayer.apply(decoderOutputs) as tf.SymbolicTensor

  const decoderModel = tf.model({
    inputs: [decoderInput, decoderStateInput],
    outputs: [decoderDenseOutputs, decoderStateOutput],
  })

  return decoderModel
}
