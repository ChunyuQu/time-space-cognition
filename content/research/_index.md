---
title: Research Program
description: 'How the past shapes perception and decisions, studied with computational models, neuroimaging, and data from multiple species'
subtitle: How the past shapes perception and decisions, studied with computational models, neuroimaging, and data from multiple species

intro:
  - 'Perception is not a snapshot. What we saw and judged a moment ago biases what we perceive now — a phenomenon called <strong>serial dependence</strong>. My research asks how the brain carries this history: which parts of the past are kept, how strongly they weigh on the present, and how their readout is controlled by the current task.'
  - 'I build generative models — Kalman filters and latent dynamics models in which dependence on history emerges from Bayesian inference — and test their predictions in human behavior, fMRI and EEG, in mouse calcium imaging, and in brain-computer interfaces that run in real time.'

stages:
  - stage: Integration
    description: New input is weighted by its reliability
    status: '✅ Submitted'
  - stage: Retention
    description: Which parts of the past are kept — and for how long
    status: '🔧 In progress'
  - stage: Readout
    description: How the task at hand controls what history is expressed
    status: '✍️ Writing'
  - stage: Monitoring
    description: How confidence tracks the uncertainty of the readout
    status: '📊 Data in hand'

anchors:
  - title: 'Computational Model: The Kalman Filter'
    body: 'The core working model has three latent states: a perception of the current input, a memory that integrates past observations, and a decision that generates the response. In this model each update is weighted by its precision, which reproduces serial dependence across sensory and contextual transitions, without any ad hoc learning rule.'
    list:
      - '<strong>Perception state:</strong> current sensory input'
      - '<strong>Memory state:</strong> integration of past information'
      - '<strong>Decision state:</strong> response generation'
    image: /images/kalman-filter-3state.png
    image_alt: '3-State Kalman Filter Architecture'
    frame: light

  - title: 'Across Species: Mouse Calcium Imaging'
    body: 'Do neurons in mouse cortex carry the same history from trial to trial? With the Max Planck Institute for Biological Intelligence, I analyze calcium imaging and EMG data — motion correction, ROI segmentation, and neuronal activity visualization — to test how sequential effects look at the neural level, across species.'
    note: '<strong>Status:</strong> submitted to Neuron | <strong>Partner:</strong> MPI for Biological Intelligence'
    image: /images/mouse-experiment.jpg
    image_alt: 'Mouse Calcium Imaging'
    frame: cover
    mobile_first: true

  - title: 'Application: Brain-Computer Interfaces'
    body: 'The same inference runs in real time. During my internship at the Munich Institute of Biomedical Engineering (TUM), I built hybrid fNIRS + EEG brain-computer interfaces with visual stimulation in virtual reality — decoding neural signals on the fly, where every millisecond of history matters.'
    note: '<strong>Internship:</strong> MIBE, Technical University of Munich | <strong>Duration:</strong> 05/2024 – 11/2024'
    image: /images/bci-internship.webp
    image_alt: 'Brain-Computer Interface Research'
    frame: cover
---
