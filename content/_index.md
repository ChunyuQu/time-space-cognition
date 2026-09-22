---
# Leave the homepage title empty to use the site title
title: ''
summary: ''
date: 2022-10-24
type: landing

sections:
  - block: resume-biography-3
    content:
      # Choose a user profile to display (a folder name within `content/authors/`)
      username: me
      text: ''
      button:
        text: Publications
        url: publications/
      headings:
        about: ''
        education: ''
        interests: ''
    design:
      # Flat surface, matching the light slate background
      background:
        gradient_mesh:
          enable: false

      # Name heading sizing to accommodate long or short names
      name:
        size: md # Options: xs, sm, md, lg (default), xl

      # Avatar customization
      avatar:
        size: medium # Options: small (150px), medium (200px, default), large (320px), xl (400px), xxl (500px)
        shape: rounded # Options: circle (default), square, rounded
  - block: markdown
    content:
      title: 'My Research'
      subtitle: ''
      text: |-
        Perception is not a snapshot. What we saw and judged a moment ago biases what we perceive now. This is serial dependence. My research asks how the brain carries that history: which parts of the past are kept, how strongly they weigh on the present, and how the current task controls what gets read out.

        I build generative models, Kalman filters and latent dynamics models in which dependence on history emerges from Bayesian inference, and test their predictions in human behavior, in fMRI and EEG, in mouse calcium imaging, and in brain-computer interfaces that run in real time.

        If you work on related questions, I would be glad to hear from you.
    design:
      columns: '1'
  - block: collection
    id: papers
    content:
      title: Featured Publications
      filters:
        folders:
          - publications
        featured_only: true
    design:
      view: article-grid
      columns: 2
  - block: collection
    content:
      title: Recent Publications
      text: ''
      filters:
        folders:
          - publications
        exclude_featured: false
    design:
      view: citation
---
