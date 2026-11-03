/** Leviathan mark: three hash-chained KV blocks; the last one holds the prefix. */
export function Mark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <path d="M2 16 H30" fill="none" stroke="currentColor" strokeWidth="2" />
      <rect x="3" y="10" width="7.5" height="12" fill="var(--paper)" stroke="currentColor" strokeWidth="2" />
      <rect x="12.25" y="10" width="7.5" height="12" fill="var(--paper)" stroke="currentColor" strokeWidth="2" />
      <rect x="21.5" y="10" width="7.5" height="12" fill="currentColor" stroke="currentColor" strokeWidth="2" />
    </svg>
  );
}
