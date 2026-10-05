export type AppTheme = 'default' | 'cuda-night' | 'next-light';

export const THEME_STORAGE_KEY = 'app_theme';

export interface ThemeOption {
  id: AppTheme;
  name: string;
  icon: string;
  badge: string;
  desc: string;
}

export const THEME_OPTIONS: ThemeOption[] = [
  { id: 'default', name: 'Default Dark', icon: '🌑', badge: 'Classic', desc: 'Original deep slate/obsidian' },
  { id: 'cuda-night', name: 'CUDA Night', icon: '🟢', badge: 'NVIDIA', desc: 'NVIDIA carbon & light green-white' },
  { id: 'next-light', name: 'Next.js Light', icon: '☀️', badge: 'Studio', desc: 'Crisp studio white & charcoal' },
];

export function getStoredTheme(): AppTheme {
  if (typeof window === 'undefined') return 'default';
  try {
    const saved = localStorage.getItem(THEME_STORAGE_KEY) as AppTheme;
    if (saved === 'default' || saved === 'cuda-night' || saved === 'next-light') {
      return saved;
    }
  } catch {}
  return 'default';
}

export function applyTheme(theme: AppTheme) {
  if (typeof document === 'undefined') return;
  const root = document.documentElement;
  root.setAttribute('data-theme', theme);

  if (theme === 'next-light') {
    root.classList.remove('dark');
    root.classList.add('light');
  } else {
    root.classList.add('dark');
    root.classList.remove('light');
  }

  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {}

  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('app-theme-changed', { detail: { theme } }));
  }
}
