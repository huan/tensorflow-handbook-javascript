#!/usr/bin/env ts-node

// tslint:disable:no-shadowed-variable
import test  from 'blue-tape'

import { Tokenizer } from './tokenizer'

test('tokenize() without args', async t => {
  const tokenizer = new Tokenizer()

  const TEXT = 'Hello, World!'
  const EXPECTED_TOKENS = ['Hello', ',', 'World', '!']

  const tokenList = [...tokenizer.tokenize(TEXT)]
  t.deepEqual(tokenList, EXPECTED_TOKENS, 'should get the token by delimiter')
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

  const TEXT = 'Hello!'
  const EXPECTED_TOKENS = ['H', 'e', 'l' ,'l', 'o', '!']

  const tokenList = [...tokenizer.tokenize(TEXT)]
  t.deepEqual(tokenList, EXPECTED_TOKENS, 'should get the token by char')
})
