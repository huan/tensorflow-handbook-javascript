import { Tokenizer } from './tokenizer'

interface VocabularyJson {
  tokenIndice: string,
  maxSeqLength: number,
  size: number,
  tokenizer: string,
}

export class Vocabulary {
  public size: number
  public maxSeqLength: number

  public tokenizer: Tokenizer

  public tokenIndice: Map<string, number>
  public indiceToken: Map<number, string>

  constructor () {
    this.tokenizer = new Tokenizer()

    this.tokenIndice = new Map<string, number>()
    this.indiceToken = new Map<number, string>()

    this.size = 1 // Including the reserved 0
    this.maxSeqLength = 0
  }

  public fitToken(token: string): void {
    if (!this.tokenIndice.has(token)) {

      this.tokenIndice.set(token, this.size)
      this.indiceToken.set(this.size, token)

      this.size++
    }
  }

  public fitText(text: string): void {
    const tokenList = [...this.tokenizer.tokenize(text)]

    if (tokenList.length > this.maxSeqLength) {
      this.maxSeqLength = tokenList.length
    }
    for (const token of tokenList) {
      this.fitToken(token)
    }
  }

  /**
   * Get the sequence of the text
   * @param text  input text string
   * @param length
   *  0 means the actual text length(after tokenize)
   *  -1 means padding 0 to the maxSeqLength
   */
  public sequenize (
    text: string,
    length = 0,
  ): number[] {
    const tokenList = [...this.tokenizer.tokenize(text)]
    const indiceList = tokenList.map(token => this.indice(token))

    if (length === -1) {
      indiceList.length = this.maxSeqLength
      if (this.maxSeqLength > tokenList.length) {
        indiceList.fill(0, tokenList.length)
      }
    }

    return indiceList
  }

  public token(indice: number): string {
    if (this.indiceToken.has(indice)) {
      return this.indiceToken.get(indice) as string
    }
    throw new Error(`token not found for indice: ${indice}`)
  }

  public indice (token: string): number {
    if (this.tokenIndice.has(token)) {
      return this.tokenIndice.get(token) as number
    }
    throw new Error(`indice not found for token: "${token}"`)
  }

  public toJSON (): VocabularyJson {
    const vocabularyJson: VocabularyJson = {
      tokenIndice: JSON.stringify([...this.tokenIndice]),
      maxSeqLength: this.maxSeqLength,
      size: this.size,
      tokenizer: JSON.stringify(this.tokenizer),
    }

    return vocabularyJson
  }

  public static fromJSON (json: string | object): Vocabulary {
    let vocabularyJson: VocabularyJson

    if (json instanceof Object) {
      vocabularyJson = json as VocabularyJson
    } else {
      vocabularyJson = JSON.parse(json)
    }

    const voc = new Vocabulary()

    voc.size = vocabularyJson.size
    voc.maxSeqLength = vocabularyJson.maxSeqLength
    voc.tokenizer = Tokenizer.fromJSON(vocabularyJson.tokenizer)

    // TODO: fix the map parse
    const tokenIndiceTuple = JSON.parse(vocabularyJson.tokenIndice) as [string, number][]
    voc.tokenIndice = new Map([...tokenIndiceTuple])

    const indiceTokenTuple = tokenIndiceTuple.map(([token, indice]) => [indice, token]) as [number, string][]
    voc.indiceToken = new Map([...indiceTokenTuple])

    return voc
  }
}
