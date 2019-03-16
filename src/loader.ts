import fs from 'fs'

import { tf }         from './config'
import { Vocabulary } from './vocabulary'

interface LoaderStore {
  encoder: tf.LayersModel,
  decoder: tf.LayersModel,
  model: tf.LayersModel,
  inputVoc: Vocabulary,
  outputVoc: Vocabulary,
}

export class Loader {
  public static async save(
    store: LoaderStore,
    workDir: string,
  ): Promise<void> {
    const {
      encoder,
      decoder,
      model,
      inputVoc,
      outputVoc,
    } = store

    const urlWorkDir = 'file://' + workDir
    await encoder.save(urlWorkDir + '/' + 'encoder')
    await decoder.save(urlWorkDir + '/' + 'decoder')
    await model.save(urlWorkDir + '/' + 'model')

    fs.writeFileSync(workDir + '/inputVoc.json', JSON.stringify(inputVoc))
    fs.writeFileSync(workDir + '/outputVoc.json', JSON.stringify(outputVoc))
  }

  public static async load(
    workDir: string,
  ): Promise<LoaderStore> {
    const urlWorkDir = 'file://' + workDir

    const encoder = await tf.loadLayersModel(urlWorkDir + '/' + 'encoder/model.json')
    const decoder = await tf.loadLayersModel(urlWorkDir + '/' + 'decoder/model.json')
    const model = await tf.loadLayersModel(urlWorkDir + '/' + 'model/model.json')

    const inputVocJsonText = fs.readFileSync(workDir + '/inputVoc.json').toString()
    const outputVocJsonText = fs.readFileSync(workDir + '/outputVoc.json').toString()

    const inputVoc = Vocabulary.fromJSON(inputVocJsonText)
    const outputVoc = Vocabulary.fromJSON(outputVocJsonText)

    return {
      encoder,
      decoder,
      model,
      inputVoc,
      outputVoc,
    }
  }
}
