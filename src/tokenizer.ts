type TokenizeMode = 'char' | 'delimiter'

interface TokenizerOptions {
  mode: TokenizeMode,
  delimiter?: RegExp,
}

const DEFAULT_DELIMITER_REGEXP = /\s+/

interface TokenizerJson {
  mode: string,
  delimiter?: string
}

export class Tokenizer {
  public readonly mode: TokenizeMode
  public readonly delimiter?: RegExp

  constructor (options?: TokenizerOptions) {
    const {
      mode,
      delimiter,
    } = {
      mode: 'delimiter',
      // very tricky for setting RegExp for delimiter, remember do add unit test when modify this.
      delimiter: DEFAULT_DELIMITER_REGEXP,
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

  public toJSON (): TokenizerJson {
    const tokenizerJson: TokenizerJson = {
      mode: this.mode,
    }

    if (this.delimiter) {
      const delimiter = this.delimiter
                            .toString()
                            // get rid of the start '/' and the end '/'
                            .slice(1, -1)
      tokenizerJson.delimiter = delimiter
    }

    return tokenizerJson
  }

  public static fromJSON (json: string | object): Tokenizer {
    let jsonObj: TokenizerJson

    if (json instanceof Object) {
      jsonObj = json as TokenizerJson
    } else {
      jsonObj = JSON.parse(json)
    }

    const options: TokenizerOptions = {
      mode: jsonObj.mode as TokenizeMode,
    }

    if (jsonObj.delimiter) {
      options.delimiter = new RegExp(jsonObj.delimiter)
    }

    return new Tokenizer(options)
  }
}
