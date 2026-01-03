import type { DocsLayoutProps } from 'fumadocs-ui/layout';

export const baseOptions: Partial<DocsLayoutProps> = {
  nav: {
    title: 'lux-fhe',
  },
  links: [
    {
      text: 'Documentation',
      url: '/docs',
      active: 'nested-url',
    },
    {
      text: 'GitHub',
      url: 'https://github.com/luxfi/fhe',
    },
  ],
};
