
type TokenizeMode = 'char' | 'delimiter'

interface TokenizerOptions {
  mode: TokenizeMode,
  delimiter?: string | RegExp,
}

export class Tokenizer {
  private readonly mode: TokenizeMode
  private readonly delimiter?: string | RegExp

  constructor (options?: TokenizerOptions) {
    const {
      mode,
      delimiter,
    } = {
      mode: 'delimiter',
      // very tricky for setting RegExp for delimiter, remember do add unit test when modify this.
      delimiter: /\s|\b/,
      ...options
    } as TokenizerOptions

    this.mode = mode
    this.delimiter = delimiter
  }

  public *tokenize (text: string): IterableIterator<string> {
    // console.log('tokenize mode:', this.mode, text)
    switch (this.mode) {
      case 'delimiter':
        for (const token of this.tokenizeByDelimiter(text, this.delimiter)) {
          yield token
        }
        break

      case 'char':
        for (const token of this.tokenizeByChar(text)) {
          yield token
        }
        break

      default:
        throw new Error('not supported mode: ' + this.mode)
    }
  }

  private *tokenizeByChar (text: string): IterableIterator<string> {
    for (const char of text) {
      yield char
    }
  }

  private *tokenizeByDelimiter (
    text: string,
    delimiter?: string | RegExp,
  ): IterableIterator<string> {
    if (!delimiter) {
      throw new Error('no delimiter set')
    }
    // let trickyText = text.replace(/\t/, ' \t ' )
    // trickyTest = text.replace(/\n/, ' \n ')
    for (const token of text.split(delimiter)) {
      if (token.length) {
        yield token
      }
    }
  }
}
