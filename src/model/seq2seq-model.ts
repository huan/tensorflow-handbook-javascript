import { tf } from '../config'

import { Vocabulary } from '../vocabulary'

interface Seq2seqModelOptions {
  encoderEmbeddingLayer: tf.layers.Layer,
  encoderRnnLayer: tf.layers.Layer,
  decoderDenseLayer: tf.layers.Layer,
  decoderEmbeddingLayer: tf.layers.Layer,
  decoderRnnLayer: tf.layers.Layer,
  inputVoc: Vocabulary,
  outputVoc: Vocabulary,
}

/**
 * Model for training
 */
export function getSeq2seqModel(
  options: Seq2seqModelOptions,
): tf.LayersModel {
  const {
    encoderEmbeddingLayer,
    encoderRnnLayer,
    decoderDenseLayer,
    decoderEmbeddingLayer,
    decoderRnnLayer,
    inputVoc,
    outputVoc,
  } = options

  const inputs = tf.layers.input({
    shape: [inputVoc.maxSeqLength],
    name: 'seq2seqInputs',
  })

  const encoderEmbedding = encoderEmbeddingLayer.apply(inputs) as tf.Tensor<tf.Rank.R3>

  const [, encoderState] = encoderRnnLayer.apply(encoderEmbedding) as tf.SymbolicTensor[]

  const decoderInputs = tf.layers.input({
    shape: [outputVoc.maxSeqLength],
    name: 'seq2seqDecoderInputs',
  })

  const decoderEmbedding = decoderEmbeddingLayer.apply(decoderInputs) as tf.SymbolicTensor

  const [decoderOutputs,] = decoderRnnLayer.apply(
    [decoderEmbedding, encoderState],
    {
      returnSequences: true,
      returnState: true,
    },
  ) as tf.SymbolicTensor[]

  const decoderTargets = decoderDenseLayer.apply(decoderOutputs) as tf.SymbolicTensor

  const seq2seqModel = tf.model({
    inputs: [inputs, decoderInputs],
    outputs: decoderTargets,
    name: 'seq2seqModel',
  })

  return seq2seqModel
}
