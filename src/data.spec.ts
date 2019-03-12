#!/usr/bin/env ts-node

// tslint:disable:no-shadowed-variable
import test  from 'blue-tape'

import { getDataset } from './data'

const FIXTURE_DATASET_CSV_FILE = 'file://tests/fixtures/dataset.csv'

test('getDataset()', async t => {
  const {
    seq2seqDataset: dataset,
    inputVoc: srcVoc,
    outputVoc: dstVoc,
    size,
  } = await getDataset(FIXTURE_DATASET_CSV_FILE)
  // } = await getDataset('file://dist/fra.txt', 10)

  console.log('srcVoc', srcVoc.size)
  console.log('srcVoc', JSON.stringify([...srcVoc.tokenIndice]))
  console.log('dstVoc', dstVoc.size)
  console.log('dstVoc', JSON.stringify([...dstVoc.tokenIndice]))
  console.log('size', size)

  await dataset.forEachAsync(value => {
    console.log('encoderInput')
    value.encoderInput.print()
    console.log('decoderInput')
    value.decoderInput.print()
    console.log('decoderOutput')
    value.decoderTarget.print()
  })

  t.ok('test')
})
