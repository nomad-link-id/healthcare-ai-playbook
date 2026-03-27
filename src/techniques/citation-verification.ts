export function verifyCitations(response: string, docs: any[]) {
  const cited = [...response.matchAll(/\[(\d+)\]/g)].map(m => parseInt(m[1]));
  const phantoms = cited.filter(id => id < 1 || id > docs.length);
  let cleaned = response;
  phantoms.forEach(id => { cleaned = cleaned.replace(new RegExp(`\\s*\\[${id}\\]`, 'g'), ''); });
  return { cleaned, verified: cited.length - phantoms.length, removed: phantoms.length };
}
