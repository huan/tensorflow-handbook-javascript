import {
  tf,
  START_TOKEN,
  END_TOKEN,
}                     from '../config'
import { Vocabulary } from '../vocabulary'
import { vectorizeInput } from '../data'

export async function seq2seqDecoder (
  input: string,
  encoderModel: tf.Model,
  decoderModel: tf.Model,
  inputVoc: Vocabulary,
  outputVoc: Vocabulary,
): Promise<string> {
  const inputSeq = vectorizeInput(input, inputVoc)

  const batchedInputSeq = inputSeq.expandDims(0)
  let state = encoderModel.predict(batchedInputSeq) as tf.Tensor<tf.Rank.R2>

  // let state = encoderModel.predictOnBatch(inputSeq) as tf.Tensor<tf.Rank.R2>

  let decoderInput = outputVoc.indice(START_TOKEN)

  let decodedTensor: tf.Tensor<tf.Rank.R3>
  let decodedToken: string
  let decodedSentence = ''

  do {
    [decodedTensor, state] = decoderModel.predict([
      tf.tensor([decoderInput]),
      state,
    ]) as [
      tf.Tensor<tf.Rank.R3>,
      tf.Tensor<tf.Rank.R2>,
    ]

    let decodedIndice = await decodedTensor
                                .squeeze()
                                .argMax()
                                .array() as number

    if (decodedIndice === 0) {
      // 0 for padding, should be treated as END
      decodedToken = END_TOKEN
    } else {
      decodedToken = outputVoc.token(decodedIndice)
    }

    decodedSentence += decodedToken

    // save decoded data for next time step
    decoderInput = decodedIndice
  } while (
       decodedToken !== END_TOKEN
    && decodedSentence.length < outputVoc.size
  )

  return decodedSentence
}
