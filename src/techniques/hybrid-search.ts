export function hybridSearch(query: string, corpus: any[], config: { threshold: number }) {
  return corpus.filter(doc => doc.score >= config.threshold);
}
