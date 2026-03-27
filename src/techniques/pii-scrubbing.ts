export function scrubPII(text: string): string {
  return text.replace(/\d{3}\.\d{3}\.\d{3}-\d{2}/g, '[CPF_REDACTED]')
    .replace(/\d{2}\.\d{3}\.\d{3}\/\d{4}-\d{2}/g, '[CNPJ_REDACTED]');
}
