---
title: Research Program
type: landing
date: 2025-10-01

sections:
  - block: markdown
    content:
      title: What I Study
      text: |-
        Perception is not a snapshot. What we saw and judged a moment ago biases what we perceive now. This is serial dependence. My research asks how the brain carries that history: which parts of the past are kept, how strongly they weigh on the present, and how the current task controls what gets read out.

        I build generative models, Kalman filters and latent dynamics models in which dependence on history emerges from Bayesian inference, and test their predictions in human behavior, in fMRI and EEG, in mouse calcium imaging, and in brain-computer interfaces that run in real time.
    design:
      columns: '1'

  - block: markdown
    content:
      title: Research Framework
      subtitle: Four stages of one modeling pipeline
      text: |-
        | Stage | Question | Status |
        | --- | --- | --- |
        | Integration | New input is weighted by its reliability | Submitted |
        | Retention | Which parts of the past are kept, and for how long | In progress |
        | Readout | How the task at hand controls what history is expressed | Writing |
        | Monitoring | How confidence tracks the uncertainty of the readout | Data in hand |
    design:
      columns: '1'

  - block: markdown
    content:
      title: 'Computational Model: The Kalman Filter'
      text: |-
        The core working model has three latent states: a perception of the current input, a memory that integrates past observations, and a decision that generates the response. Each update is weighted by its precision, which reproduces serial dependence across sensory and contextual transitions without an ad hoc learning rule.

        1. Perception state: current sensory input
        2. Memory state: integration of past information
        3. Decision state: response generation

        ![Three-state Kalman filter architecture](/images/kalman-filter-3state.png)
    design:
      columns: '1'

  - block: markdown
    content:
      title: 'Across Species: Mouse Calcium Imaging'
      text: |-
        Do neurons in mouse cortex carry the same history from trial to trial? With the Max Planck Institute for Biological Intelligence, I analyze calcium imaging and EMG data, covering motion correction, ROI segmentation, and neuronal activity visualization, to test how sequential effects look at the neural level across species.

        Partner: Max Planck Institute for Biological Intelligence. Status: submitted to Neuron.

        ![Mouse calcium imaging setup](/images/mouse-experiment.jpg)
    design:
      columns: '1'

  - block: markdown
    content:
      title: 'Application: Brain-Computer Interfaces'
      text: |-
        The same inference runs in real time. During my internship at the Munich Institute of Biomedical Engineering (TUM), I built hybrid fNIRS and EEG brain-computer interfaces with visual stimulation in virtual reality, decoding neural signals on the fly, where every millisecond of history matters.

        Internship: MIBE, Technical University of Munich. Duration: 05/2024 to 11/2024.

        ![Brain-computer interface research](/images/bci-internship.webp)
    design:
      columns: '1'
---
