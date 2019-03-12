export class Tokenizer {
  constructor () {
  }

  public *tokenize (text: string): IterableIterator<string> {
    for (const char of text) {
      yield char
    }
  }
}
