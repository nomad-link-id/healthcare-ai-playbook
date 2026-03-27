const EMERGENCY_PATTERNS = ['chest pain', 'stroke', 'suicide', 'heart attack', 'cannot breathe'];
export function detectEmergency(query: string): { isEmergency: boolean; pattern?: string } {
  const found = EMERGENCY_PATTERNS.find(p => query.toLowerCase().includes(p));
  return { isEmergency: !!found, pattern: found };
}
