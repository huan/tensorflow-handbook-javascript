import path from 'path'
import fs from 'fs'

import { Vocabulary } from './vocabulary'

const META_DATA_JSON_FILE_NAME = 'metadata.json'

export async function saveMetadata (
  inputVoc: Vocabulary,
  outputVoc: Vocabulary,
  workDir: string,
) {
  console.log('Number of unique input tokens:', inputVoc.size)
  console.log('Number of unique output tokens:', outputVoc.size)
  console.log('Max sequence length for inputs:', inputVoc.maxSeqLength)
  console.log('Max sequence length for outputs:', outputVoc.maxSeqLength)

  // Save the token indices to file.
  const metadataJsonPath = path.join(
    workDir,
    META_DATA_JSON_FILE_NAME,
  )

  const metadata = {
    'inputVoc': inputVoc,
    'outputVoc': outputVoc,
  }

  fs.writeFileSync(metadataJsonPath, JSON.stringify(metadata))
  console.log('Saved metadata at: ', metadataJsonPath)
}

export async function loadMetadata (
  workDir: string,
) {
  const jsonText = fs.readFileSync(
    path.join(workDir, META_DATA_JSON_FILE_NAME),
  ).toString()

  const json = JSON.parse(jsonText)

  const srcVoc = Vocabulary.fromJSON(json.srcVoc)
  const dstVoc = Vocabulary.fromJSON(json.dstVoc)

  return {
    srcVoc,
    dstVoc,
  }
}
