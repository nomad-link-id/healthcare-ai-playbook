export function routeByComplexity(query: string, models: { id: string; maxComplexity: number }[]) {
  const score = Math.min(1, query.split(' ').length / 50);
  return models.find(m => m.maxComplexity >= score) || models[models.length - 1];
}
