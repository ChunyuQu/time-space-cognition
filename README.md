# time-space-cognition.com

Personal academic website of Chunyu Qu. Built with [Hugo](https://gohugo.io/) and
[Tailwind CSS](https://tailwindcss.com/), deployed to GitHub Pages.

The site is plain Hugo: no theme, no template framework. The layouts in `layouts/`
are the source of truth for markup, and the design is a direct port of the previous
personal site, so the two render identically.

## Requirements

- Hugo extended, v0.166.0 or later
- Node.js 20 or later

## Local development

```bash
npm install
npm run serve          # compiles CSS, then serves on http://localhost:1313
```

`npm run css` compiles Tailwind once. `npm run css:watch` rebuilds it while you work.
The generated stylesheet is `assets/css/tailwind.css` and is not committed.

## Build

```bash
npm run build          # npm run css && hugo --gc --minify
```

## Where content lives

| What | Where |
| --- | --- |
| Site title, description, role, tagline | `hugo.yaml` |
| Publications (all of them, one list) | `data/publications.yaml` |
| Home page: hero, selected research, research focus | `content/_index.md` |
| Research program: intro, four stages, three anchors | `content/research/_index.md` |
| Publications page: groups and profile links | `content/publications/_index.md` |
| Projects: three directions and their projects | `content/projects/_index.md` |
| Experience: profile, education, roles, awards, supervisors, skills | `content/experience/_index.md` |
| Contact: intro, research connections, profiles, affiliation | `content/contact/_index.md` |
| Reading Notes | `content/reading/` |
| Modeling Notes | `content/modeling/` |
| Images | `static/images/` |

### Adding a publication

Append an entry to `data/publications.yaml`:

```yaml
- id: short-slug
  title: 'Full paper title'
  authors: 'Qu, C., & Shi, Z.'
  year: 2026
  venue: bioRxiv
  role: First Author        # or Coauthor
  status: Preprint          # Preprint | Published | Submitted
  paper: https://doi.org/...
  code: https://github.com/...   # optional
```

The publications page groups entries by `status`. On the home page, an entry can be
featured by referencing its `id` under `selected` in `content/_index.md`, and on the
projects page by setting `publication_id` on a project.

### Adding a note

Create a markdown file in `content/reading/` or `content/modeling/`:

```markdown
---
title: 'Note title'
date: 2026-09-22
summary: 'One line shown in the list.'
---

Body text.
```

## Layouts

```
layouts/
  _default/baseof.html        page shell: head, navbar, main, footer
  _default/list.html          generic section page (Reading / Modeling Notes)
  _default/single.html        a single note
  index.html                  home page
  research/list.html          research program
  publications/list.html      publications, grouped by status
  projects/list.html          projects by research direction
  experience/list.html        education, roles, awards, supervisors, skills
  contact/list.html           contact page with the reveal-on-click email
  partials/head.html          meta tags, stylesheet, JSON-LD
  partials/navbar.html        sticky top navigation
  partials/footer.html        four-column footer
  partials/jsonld.html        schema.org metadata
  robots.txt                  allows search and AI crawlers explicitly
```

## Deployment

Pushing to `main` runs `.github/workflows/deploy.yml`, which installs dependencies,
compiles Tailwind, builds with Hugo, and publishes `public/` to GitHub Pages.
`actions/configure-pages` supplies the base URL, so the custom domain and the
`*.github.io` address both resolve correctly.

## Notes on discovery

The previous site blocked AI crawlers and shipped no structured data. This one does
the opposite: `robots.txt` allows search and AI crawlers by name, every page carries
JSON-LD, and Hugo generates a sitemap and RSS feed.
