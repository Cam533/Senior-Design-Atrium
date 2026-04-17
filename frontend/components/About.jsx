import { Link } from 'react-router-dom'
import { aboutCopy } from '../content/aboutCopy'
import './About.css'

const valueIcons = {
  blue: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
    </svg>
  ),
  green: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2M9 11a4 4 0 100-8 4 4 0 000 8zM23 21v-2a4 4 0 00-3-3.87M16 3.13a4 4 0 010 7.75" />
    </svg>
  ),
  indigo: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
    </svg>
  ),
}

function iconToneClass(tone) {
  if (tone === 'green') return 'about-value-icon about-value-icon--green'
  if (tone === 'indigo') return 'about-value-icon about-value-icon--indigo'
  return 'about-value-icon about-value-icon--blue'
}

export default function About() {
  const { banner, hero, mission, values, approach, built, cta } = aboutCopy
  const gallery = approach?.images?.filter((img) => img?.src) ?? []
  const builtItems = built?.items?.filter((s) => String(s).trim()) ?? []

  return (
    <div className="about-page">
      <section
        className={banner?.image ? 'about-hero' : 'about-hero about-hero--fallback'}
        aria-label="About"
      >
        {banner?.image ? (
          <img className="about-hero-img" src={banner.image} alt={banner.alt || ''} />
        ) : null}
        <div className="about-hero-overlay">
          <div className="about-hero-inner">
            <h1 className="about-hero-title">{hero.title}</h1>
            <p className="about-hero-tagline">{hero.tagline}</p>
          </div>
        </div>
      </section>

      <section className="about-section">
        <div className="about-shell">
          <div className="about-mission-grid">
            <div>
              <h2 className="about-h2">{mission?.title}</h2>
              {(mission?.paragraphs ?? []).map((p, i) => (
                <p key={i} className="about-text">
                  {p}
                </p>
              ))}
            </div>
          </div>
        </div>
      </section>

      {values?.items?.length > 0 && (
        <section className="about-section about-section--muted">
          <div className="about-shell">
            <h2 className="about-values-title">{values.title}</h2>
            <div className="about-values-grid">
              {values.items.map((item, i) => (
                <div key={i} className="about-value-card">
                  <div className={iconToneClass(item.tone)}>
                    {valueIcons[item.tone] || valueIcons.blue}
                  </div>
                  <h3>{item.title}</h3>
                  <p>{item.text}</p>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      <section className="about-section about-section--approach">
        <div className="about-shell">
          <h2 className="about-values-title">{approach?.title}</h2>
          <p className="about-approach-intro">{approach?.intro}</p>
          {(approach?.sourceLinks ?? []).length > 0 && (
            <p className="about-approach-source">
              Data via{' '}
              {(approach.sourceLinks ?? []).map((link, i) => (
                <span key={link.url || i}>
                  <a
                    className="about-link"
                    href={link.url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {link.label}
                  </a>
                  {i < approach.sourceLinks.length - 1 ? ', ' : ''}
                </span>
              ))}
            </p>
          )}
          {gallery.length > 0 && (
            <div className="about-gallery">
              {gallery.map((img, i) => (
                <img key={i} src={img.src} alt={img.alt || ''} loading="lazy" />
              ))}
            </div>
          )}
        </div>
      </section>

      {built && builtItems.length > 0 && (
        <section className="about-section about-section--muted">
          <div className="about-shell">
            <h2 className="about-values-title">{built.title}</h2>
            <div className="about-built-list">
              {builtItems.map((item, i) => (
                <div key={i} className="about-built-row">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  <span>{item}</span>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      <section className="about-cta">
        <div className="about-cta-inner">
          <h2>{cta?.title}</h2>
          <p>{cta?.body}</p>
          <Link to={cta?.buttonPath || '/'} className="about-cta-btn">
            {cta?.buttonLabel}
          </Link>
        </div>
      </section>

      <div className="about-page-footer-spacer" />
    </div>
  )
}
