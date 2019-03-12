import { tf } from './config'

import { Vocabulary } from './vocabulary'

export function createLayer (
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
      units: latentDim,
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
  } = createLayer(
    inputVoc,
    outputVoc,
    latentDim,
  )

  const model = getModel({
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
    model,
    encoderModel,
    decoderModel,
  }
}

interface ModelOptions {
  encoderEmbeddingLayer: tf.layers.Layer,
  encoderRnnLayer: tf.layers.Layer,
  decoderDenseLayer: tf.layers.Layer,
  decoderEmbeddingLayer: tf.layers.Layer,
  decoderRnnLayer: tf.layers.Layer,
  inputVoc: Vocabulary,
  outputVoc: Vocabulary,
}

function getModel(
  {
    encoderEmbeddingLayer,
    encoderRnnLayer,
    decoderDenseLayer,
    decoderEmbeddingLayer,
    decoderRnnLayer,
    inputVoc,
    outputVoc,
  }: ModelOptions,
): tf.Model {
  /**
   * Model
   */
  const inputs = tf.layers.input({
    shape: [inputVoc.maxSeqLength],
    name: 'modelInputs',
  })

  const encoderEmbedding = encoderEmbeddingLayer.apply(inputs)
  const [, encoderState] = encoderRnnLayer.apply(encoderEmbedding) as tf.SymbolicTensor[]

  const decoderInputs = tf.layers.input({
    shape: [outputVoc.size],
    name: 'decoderInputs',
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

  const model = tf.model({
    inputs: [inputs, decoderInputs],
    outputs: decoderTargets,
    name: 'model',
  })

  return model
}

interface EncoderModelOptions {
  encoderEmbeddingLayer: tf.layers.Layer,
  encoderRnnLayer: tf.layers.Layer,
  inputVoc: Vocabulary,
}

function getEncoderModel (
  {
    encoderEmbeddingLayer,
    encoderRnnLayer,
    inputVoc,
  }: EncoderModelOptions,
): tf.Model {
  /**
   * Encoder Model
   */
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

interface DecoderModelOptions {
  decoderEmbeddingLayer: tf.layers.Layer,
  decoderRnnLayer: tf.layers.Layer,
  decoderDenseLayer: tf.layers.Layer,
  latentDim: number,
}

function getDecoderModel (
  {
    decoderEmbeddingLayer,
    decoderRnnLayer,
    decoderDenseLayer,
    latentDim,
  }: DecoderModelOptions,
): tf.Model {
  /**
   * Decoder Model
   */
  const decoderInputs = tf.layers.input({
    shape: [1],
    name: 'decoderInputs',
  })
  const decoderStateInput = tf.layers.input({
    shape: [latentDim],
    name: 'decoderState',
  }) as tf.SymbolicTensor

  const decoderEmbedding = decoderEmbeddingLayer.apply(decoderInputs) as tf.SymbolicTensor

  const [decoderOutputs, decoderStateOutput] = decoderRnnLayer.apply(
    [decoderEmbedding, decoderStateInput],
    {
      returnState: true,
    },
  ) as tf.SymbolicTensor[]
  const decoderDenseOutputs = decoderDenseLayer.apply(decoderOutputs) as tf.SymbolicTensor

  const decoderModel = tf.model({
    inputs: [decoderInputs, decoderStateInput],
    outputs: [decoderDenseOutputs, decoderStateOutput],
  })

  return decoderModel
}
