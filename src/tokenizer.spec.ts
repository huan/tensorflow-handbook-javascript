#!/usr/bin/env ts-node

// tslint:disable:no-shadowed-variable
import test  from 'blue-tape'

import { Tokenizer } from './tokenizer'

test('tokenize()', async t => {
  const tokenizer = new Tokenizer()

  const TEXT = 'Hello , World !'
  const EXPECTED_TOKENS = ['Hello', ',', 'World', '!']

  const tokenList = [...tokenizer.tokenize(TEXT)]
  t.deepEqual(tokenList, EXPECTED_TOKENS, 'should get the token by default delimiter')
})

test('tokenizeByDelimiter()', async t => {
  const tokenizer = new Tokenizer({
    mode: 'delimiter',
    delimiter: /\s|\b/,
  })

  const TEXT_LIST = [
    'Hello... World!',
    'hi, how are you?',
  ]

  const EXPECTED_TOKENS_LIST = [
    ['Hello', '...', 'World', '!'],
    ['hi', ',', 'how', 'are', 'you', '?'],
  ]

  for (let i = 0; i < TEXT_LIST.length; i++) {
    const TEXT = TEXT_LIST[i]
    const EXPECTED_TOKENS = EXPECTED_TOKENS_LIST[i]

    const tokenList = [...tokenizer.tokenize(TEXT)]
    t.deepEqual(tokenList, EXPECTED_TOKENS, 'should get the token by delimiter #' + i)
  }
})

test('tokenizeByChar()', async t => {
  const tokenizer = new Tokenizer({
    mode: 'char',
  })

  const TEXT = 'Hello, world!'
  const EXPECTED_TOKENS = ['H', 'e', 'l' ,'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!']

  const tokenList = [...tokenizer.tokenize(TEXT)]
  t.deepEqual(tokenList, EXPECTED_TOKENS, 'should get the token by char')
})

test('toJSON()', async t => {
  const tokenizer = new Tokenizer({
    mode: 'delimiter',
    delimiter: /abc/,
  })

  const jsonText = JSON.stringify(tokenizer)
  const EXPECTED_JSON_TEXT = '{"mode":"delimiter","delimiter":"abc"}'

  t.equal(jsonText, EXPECTED_JSON_TEXT, 'should stringify to json')
})

test('fromJSON()', async t => {
  const JSON_TEXT = '{"mode":"delimiter","delimiter":"abc"}'
  const tokenizer = Tokenizer.fromJSON(JSON_TEXT)

  const EXPECTED_MODE = 'delimiter'
  const EXPECTED_DELIMITER = /abc/

  t.equal(tokenizer.mode, EXPECTED_MODE, 'should be parsed to right mode')
  t.ok(tokenizer.delimiter instanceof RegExp, 'should get a parsed delimiter as RegExp')
  t.equal(tokenizer.delimiter!.toString(), EXPECTED_DELIMITER.toString(), 'should be parsed to right delimiter')
})

test('toJSON() then fromJSON()', async t => {
  const tokenizer = new Tokenizer({
    mode: 'delimiter',
    delimiter: /abc/,
  })

  const jsonText = JSON.stringify(tokenizer)

  const newTokenizer = Tokenizer.fromJSON(JSON.parse(jsonText))

  const EXPECTED_MODE = 'delimiter'
  const EXPECTED_DELIMITER = /abc/

  t.equal(newTokenizer.mode, EXPECTED_MODE, 'should be parsed to right mode')
  t.ok(newTokenizer.delimiter instanceof RegExp, 'should get a parsed delimiter as RegExp')
  t.equal(newTokenizer.delimiter!.toString(), EXPECTED_DELIMITER.toString(), 'should be parsed to right delimiter')
})
