/** Minimal line icons + the hero suggestion list. */

const stroke = {
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.6,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
}

export const icons = {
  count: (
    <svg width="17" height="17" viewBox="0 0 24 24" {...stroke}>
      <path d="M9 4H7a2 2 0 0 0-2 2v3a2 2 0 1 1 0 4v3a2 2 0 0 0 2 2h2" />
      <path d="M17 4h-2a2 2 0 0 0-2 2v3a2 2 0 1 0 0 4v3a2 2 0 0 0 2 2h2" />
    </svg>
  ),
  rank: (
    <svg width="17" height="17" viewBox="0 0 24 24" {...stroke}>
      <path d="M4 20V10" /><path d="M10 20V4" /><path d="M16 20v-7" /><path d="M22 20H2" />
    </svg>
  ),
  group: (
    <svg width="17" height="17" viewBox="0 0 24 24" {...stroke}>
      <rect x="3" y="4" width="18" height="6" rx="1.5" />
      <rect x="3" y="14" width="18" height="6" rx="1.5" />
    </svg>
  ),
  time: (
    <svg width="17" height="17" viewBox="0 0 24 24" {...stroke}>
      <circle cx="12" cy="12" r="8.5" /><path d="M12 7.5V12l3 2" />
    </svg>
  ),
}

export const AttachIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" {...stroke}>
    <path d="M12 5v14M5 12h14" />
  </svg>
)

export const SUGGESTIONS = [
  { icon: 'rank', text: 'Show the best students' },
  { icon: 'group', text: 'Average marks by department' },
  { icon: 'count', text: 'How many students are in each department?' },
  { icon: 'time', text: 'Show students with high attendance' },
]
