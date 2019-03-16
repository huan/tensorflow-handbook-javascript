#!/usr/bin/env ts-node

// tslint:disable:no-shadowed-variable
import test  from 'blue-tape'

import { Vocabulary } from './vocabulary'

test('Vocabulary()', async t => {
  const vocabulary = new Vocabulary()

  vocabulary.fitText('Hello .')

  const EXPECTED_SIZE = 3
  const EXPECTED_MAX_LENGTH = 2
  const EXPECTED_TOKEN_INDICE = new Map([
    ['Hello', 1],
    ['.', 2],
  ])
  const EXPECTED_INDICE_TOKEN = new Map([
    [1, 'Hello'],
    [2, '.'],
  ])

  t.equal(vocabulary.size, EXPECTED_SIZE, 'should get right size')
  t.equal(vocabulary.maxSeqLength, EXPECTED_MAX_LENGTH, 'should get right seq length')
  t.deepEqual([...vocabulary.tokenIndice], [...EXPECTED_TOKEN_INDICE], 'should set the right token indice map')
  t.deepEqual([...vocabulary.indiceToken], [...EXPECTED_INDICE_TOKEN], 'should set the right indice token map')
})

test('sequenize()', async t => {
  const vocabulary = new Vocabulary()

  const TEXT = 'a b c d'

  const TEST_LIST = [
    ['a b c d', 0, [1, 2, 3, 4]],
    ['a b c', -1, [1, 2, 3, 0]],
    ['d c b a', -1, [4, 3, 2, 1]],
    ['d c b a', 0, [4, 3, 2, 1]],
    ['d c b a a', 0, [4, 3, 2, 1, 1]],
    ['d c b a a', -1, [4, 3, 2, 1]],
  ] as [string, number, number[]][]

  vocabulary.fitText(TEXT)

  for (const [i, [text, len, EXPECTED_SEQ]] of TEST_LIST.entries()) {
    const seq = vocabulary.sequenize(text, len)
    t.deepEqual(seq, EXPECTED_SEQ, 'should sequenize to right seq #' + i)
  }

})

test('toJSON()', async t => {
  const vocabulary = new Vocabulary()

  vocabulary.fitText('Hello .')

  const jsonText = JSON.stringify(vocabulary)
  const EXPECTED_JSON_TEXT = '{"tokenIndice":"[[\\"Hello\\",1],[\\".\\",2]]","maxSeqLength":2,"size":3,"tokenizer":"{\\"mode\\":\\"delimiter\\",\\"delimiter\\":\\"\\\\\\\\s+\\"}"}'

  t.equal(jsonText, EXPECTED_JSON_TEXT, 'should stringify to json')
})

test('fromJSON()', async t => {
  const JSON_TEXT = '{"tokenIndice":"[[\\"Hello\\",1],[\\".\\",2]]","maxSeqLength":2,"size":3,"tokenizer":"{\\"mode\\":\\"delimiter\\",\\"delimiter\\":\\"\\\\\\\\s+\\"}"}'
  const vocabulary = Vocabulary.fromJSON(JSON_TEXT)

  const EXPECTED_SIZE = 3
  const EXPECTED_MAXSEQLENGTH = 2
  const EXPECTED_TOKEN_INDICE = new Map([
    ['Hello', 1],
    ['.', 2],
  ])
  const EXPECTED_TOKENIZER_MODE = 'delimiter'
  const EXPECTED_TOKENIZER_DELIMITER = /\s+/

  t.equal(vocabulary.size, EXPECTED_SIZE, 'should be parsed to right size')
  t.equal(vocabulary.maxSeqLength, EXPECTED_MAXSEQLENGTH, 'should be parsed to right max length')
  t.deepEqual([...vocabulary.tokenIndice], [...EXPECTED_TOKEN_INDICE], 'shohuld set the tokenIndice map right')
  t.equal(vocabulary.tokenizer.mode, EXPECTED_TOKENIZER_MODE, 'should get the right tokenizer mode')
  t.equal(vocabulary.tokenizer.delimiter!.toString(), EXPECTED_TOKENIZER_DELIMITER.toString(), 'should set the right tokenizer delimiter')
})

test('toJSON() then fromJSON()', async t => {
  const originalVocabulary = new Vocabulary()

  originalVocabulary.fitText('Hello .')

  const jsonText = JSON.stringify(originalVocabulary)

  const vocabulary = Vocabulary.fromJSON(jsonText)

  const EXPECTED_SIZE = 3
  const EXPECTED_MAXSEQLENGTH = 2
  const EXPECTED_TOKEN_INDICE = new Map([
    ['Hello', 1],
    ['.', 2],
  ])
  const EXPECTED_TOKENIZER_MODE = 'delimiter'
  const EXPECTED_TOKENIZER_DELIMITER = /\s+/

  t.equal(vocabulary.size, EXPECTED_SIZE, 'should be parsed to right size')
  t.equal(vocabulary.maxSeqLength, EXPECTED_MAXSEQLENGTH, 'should be parsed to right max length')
  t.deepEqual([...vocabulary.tokenIndice], [...EXPECTED_TOKEN_INDICE], 'shohuld set the tokenIndice map right')
  t.equal(vocabulary.tokenizer.mode, EXPECTED_TOKENIZER_MODE, 'should get the right tokenizer mode')
  t.equal(vocabulary.tokenizer.delimiter!.toString(), EXPECTED_TOKENIZER_DELIMITER.toString(), 'should set the right tokenizer delimiter')
})
