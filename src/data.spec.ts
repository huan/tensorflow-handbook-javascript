#!/usr/bin/env ts-node

// tslint:disable:no-shadowed-variable
import test  from 'blue-tape'

import { getDataset } from './data'

const FIXTURE_DATASET_CSV_FILE = 'file://tests/fixtures/dataset.csv'

test('getDataset()', async t => {
  const {
    seq2seqDataset,
    inputVoc,
    outputVoc,
    size,
  } = await getDataset(FIXTURE_DATASET_CSV_FILE)
  // } = await getDataset('file://dist/fra.txt', 10)

  console.log('srcVoc', inputVoc.size)
  console.log('srcVoc', JSON.stringify([...inputVoc.tokenIndice]))
  console.log('dstVoc', outputVoc.size)
  console.log('dstVoc', JSON.stringify([...outputVoc.tokenIndice]))
  console.log('size', size)

  await seq2seqDataset.forEachAsync(([xs, ys]) => {
    console.log('encoderInput')
    xs.seq2seqInputs.print()
    console.log('decoderInput')
    xs.seq2seqDecoderInputs.print()
    console.log('decoderOutput')
    ys.print()
  })

  t.ok('test')
})
