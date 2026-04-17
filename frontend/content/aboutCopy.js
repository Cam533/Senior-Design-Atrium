/**
 * About page (/about) — edit copy here. Banner file: aboutBanner.js
 */

import bannerImageUrl from './aboutBanner.js'

export const aboutCopy = {
  banner: {
    image: bannerImageUrl,
    alt: 'Philadelphia skyline at sunset',
  },

  hero: {
    title: 'About Us',
    tagline:
      "Shaping tomorrow's cities through thoughtful design, data, and sustainable planning.",
  },

  mission: {
    title: 'Our mission',
    paragraphs: [
      'We believe that well-designed urban spaces and transparent land data can transform communities, improve quality of life, and support sustainable growth.',
      'Atrium combines maps, zoning-aware insight, and research so developers, planners, and residents can discover and evaluate underused urban land together.',
    ],
  },

  values: {
    title: 'Our values',
    items: [
    ],
  },

  approach: {
    title: 'Our approach',
    intro:
      'We source data from OpenDataPhilly and transform it into actionable insights by identifying underutilized land, computing spatial scores for walkability, environmental quality, and transit accessibility, and presenting it through an interactive map, while also enabling community-sourced input to increase visibility and engagement around development opportunities.',
    sourceLinks: [
      { label: 'OpenDataPhilly', url: 'https://opendataphilly.org/' },
    ],
    images: [],
  },

  /** Set to null to hide */
  built: {
    title: "What we've built",
    items: [
    ],
  },

  cta: {
    title: 'Explore the city with Atrium',
    body: 'Jump to the map to discover lots, run analysis, and keep track of places you care about.',
    buttonLabel: 'Explore the map',
    buttonPath: '/',
  },
}
