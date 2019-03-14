import {
  tf,
  START_TOKEN,
  END_TOKEN,
}                     from '../config'
import { Vocabulary } from '../vocabulary'
import { vectorizeInput } from '../data'

export async function seq2seqDecoder (
  input: string,
  encoderModel: tf.LayersModel,
  decoderModel: tf.LayersModel,
  inputVoc: Vocabulary,
  outputVoc: Vocabulary,
): Promise<string> {
  const inputSeq = vectorizeInput(input, inputVoc)

  const batchedInputSeq = inputSeq.expandDims(0)
  let state = encoderModel.predict(batchedInputSeq) as tf.Tensor<tf.Rank.R2>

  // let state = encoderModel.predictOnBatch(inputSeq) as tf.Tensor<tf.Rank.R2>

  let tokenIndice = outputVoc.indice(START_TOKEN)

  let decoderOutputs: tf.Tensor<tf.Rank.R3>
  let decodedToken: string
  let decodedTokenList = []

  do {
    const decoderInputs = tf.tensor(tokenIndice).reshape([1, 1]) as tf.Tensor<tf.Rank.R2>

    // console.log('decoderInputTensor', decoderInputs.arraySync())
    // console.log('decoderInputTensor', decoderInputs.shape)
    // console.log('state', state.shape)

    ;[decoderOutputs, state] = decoderModel.predict([
      decoderInputs,
      state,
    ]) as [
      tf.Tensor<tf.Rank.R3>,
      tf.Tensor<tf.Rank.R2>,
    ]

    // console.log('decoderOutputs', decoderOutputs.shape)

    let decodedIndice = await decoderOutputs
                                .squeeze()
                                .argMax()
                                .array() as number

    if (decodedIndice === 0) {
      // 0 for padding, should be treated as END
      decodedToken = END_TOKEN
    } else {
      decodedToken = outputVoc.token(decodedIndice)
    }

    if (decodedToken === END_TOKEN) {
      break
    } else {
      decodedTokenList.push(decodedToken)
    }

    // save decoded data for next time step
    tokenIndice = decodedIndice

  } while (decodedTokenList.length < outputVoc.maxSeqLength)

  return decodedTokenList.join(' ')
}
