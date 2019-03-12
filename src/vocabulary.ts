import { Tokenizer } from './tokenizer'

export class Vocabulary {
  public readonly tokenIndice: Map<string, number>
  public readonly indiceToken: Map<number, string>

  public maxSeqLength: number
  public size: number

  public readonly tokenizer: Tokenizer

  constructor () {
    this.tokenizer = new Tokenizer()

    this.tokenIndice = new Map<string, number>()
    this.indiceToken = new Map<number, string>()

    this.size = 1 // Including the reserved 0
    this.maxSeqLength = 0
  }

  public fitText(text: string): void {
    if (text.length > this.maxSeqLength) {
      this.maxSeqLength = text.length
    }
    for (const token of this.tokenizer.tokenize(text)) {
      if (!this.tokenIndice.has(token)) {

        this.tokenIndice.set(token, this.size)
        this.indiceToken.set(this.size, token)

        this.size++
      }
    }
  }

  public sequenize (text: string): number[] {
    const tokenList = [...this.tokenizer.tokenize(text)]
    const indiceList = tokenList.map(token => this.indice(token))
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

  public toJSON (): string {
    const tokenIndice = this.tokenIndice
    const maxSeqLength = this.maxSeqLength

    const tokenizer = this.tokenizer

    const metadata = {
      tokenIndice,
      maxSeqLength,
      tokenizer,
    }

    return JSON.stringify(metadata)
  }

  public static fromJSON (text: string) {
    console.log('TODO', text.length)
    // TODO: deserilization
  }
}
