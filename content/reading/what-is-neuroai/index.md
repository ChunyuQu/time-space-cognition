---
title: 'What exactly is NeuroAI?'
date: 2026-09-20
collection: cognitive-essays
topic: Essay
summary: 'NeuroAI is not one method. It is a two-way research program: neuroscience supplies principles for intelligence, and AI supplies models and instruments for neuroscience.'
author: 'Chunyu Qu'
cover: '/images/reading/what-is-neuroai.webp'
---

NeuroAI is easy to recognize from a conference title and difficult to define in a laboratory. A spiking chip, a vision model compared with cortex, a language model used to predict brain responses, an automated behavior pipeline, and a neural foundation model may all carry the label. If the term includes everything touching neurons and algorithms, it explains nothing.

The most useful definition is directional. NeuroAI is a two-way research program. One direction asks what principles of biological intelligence can produce more capable, efficient, adaptive artificial systems. The other uses AI models as explicit hypotheses and instruments for understanding nervous systems. A project belongs to the field when the exchange changes the scientific object, not merely because a neural network appears in the methods section.

## Before the name

The traffic is older than “NeuroAI.” McCulloch and Pitts abstracted neurons into logic. Hebb proposed learning through coactivity. Rosenblatt’s perceptron joined a biological metaphor to a trainable machine. Convolutional networks inherited ideas about receptive fields and hierarchical vision, while reinforcement learning formalized relations among reward, prediction, and action that later shaped theories of dopamine.

The influence was never a simple transfer from brain to machine. Once an algorithm existed, it became a model that could be compared with behavior and neural activity. Backpropagation was criticized as biologically implausible, yet trained networks became useful controlled systems for asking what representations a task demands.

## Zador’s research program

Anthony Zador argued that neuroscience should seek the algorithms that let animals learn rapidly, generalize, and act under constraints that current AI handles poorly. Brains operate on little power, learn continuously, exploit embodiment, and combine inherited structure with experience. Reproducing those capacities does not require copying every ion channel. It requires identifying the correct level of abstraction.

This turns biological detail into a selection problem. Which features are computational principles and which are implementation contingencies? Recurrence, sparse activity, local plasticity, multiple time scales, memory systems, active sensing, and developmental priors are candidates. Each needs an engineering test: does the principle improve learning or robustness under a defined constraint?

## AI is also changing neuroscience

The reverse direction is equally important. Deep models fit high-dimensional stimulus-response mappings. Self-supervised models provide representations learned without hand labels. Generative models describe images, movement, neural population activity, and latent dynamics. Language models can formalize hypotheses, search literature, or generate experimental code, though none of these uses turns their output into evidence automatically.

The strongest use of an AI model is not “its accuracy is high.” It is a controlled comparison. Which objective, architecture, data, or inductive bias makes internal representations align with behavior or neural recordings? What perturbation causes model and organism to fail in the same way? Prediction begins the analysis; intervention gives it explanatory force.

## Ten active directions

### Embodied and developmentally constrained learning

Agents learn through action, limited sensors, energetic cost, and a developmental curriculum. The aim is to replace static dataset learning with the closed perception-action loop that structures animal experience.

### NeuroAI robotics

Robots provide a physical test bed for active sensing, sensorimotor prediction, adaptive control, and event-driven computation. A brain-inspired controller must survive delays, noise, and changing bodies rather than only a benchmark.

### Human and machine learning

This area compares sample efficiency, curriculum effects, abstraction, transfer, and catastrophic forgetting. Human behavior is not simply a target score; detailed error patterns constrain the algorithm.

### Vision and brain models

Task-trained visual networks are compared with cortical response patterns and behavior. Current work moves beyond correlation toward out-of-distribution tests, recurrent processing, eye movements, and causal perturbation.

### Neuromorphic computation

Event-driven sensors, spiking circuits, analog devices, and local learning seek low-latency, low-power computation. Biological resemblance matters only when it yields a measurable computational advantage or a testable model.

### Neural foundation models and BCIs

Large self-supervised models attempt to transfer across people, sessions, devices, and tasks. Their central challenge is not scale alone, but heterogeneous electrodes, sampling rates, montages, and neural meanings.

### Behavior, connectomics, and automated experiments

Computer vision quantifies pose and social interaction. Connectomic methods reconstruct circuits. Closed-loop systems choose the next stimulus from incoming data. Automation changes what can be measured, but also creates new error propagation.

### Unified benchmarks for model and brain

Benchmarks compare prediction, representation, behavior, robustness, and intervention across models and organisms. A leaderboard is useful only if its tasks discriminate scientific hypotheses.

### Language, communication, and cognition

Language models create new opportunities to compare compositional structure, semantic prediction, dialogue, and brain responses. Fluency alone is a poor theory of language; dissociations and controlled generalization remain essential.

### Hypothesis generation, simulation, and governance

AI can propose experiments, simulate populations, and search models. Because the same systems can create persuasive but unsupported explanations, provenance, uncertainty, reproducibility, and governance are scientific infrastructure rather than administrative afterthoughts.

## Who enters the field

NeuroAI draws from neuroscience, psychology, machine learning, robotics, physics, mathematics, medicine, and engineering. These backgrounds carry different defaults. A neuroscientist may treat an architecture as a hypothesis about mechanism. An engineer may treat biology as a library of useful design patterns. A cognitive scientist may care most about the task and latent computation.

Collaboration fails when these levels are left implicit. A model can predict neurons without resembling their implementation. It can reproduce behavior through the wrong representation. A biologically detailed simulation can explain little if no competing mechanism is tested.

## What institutions are building

Dedicated programs now appear in universities, research institutes, funding calls, industrial laboratories, and conferences. Some emphasize brain-inspired general intelligence, some scientific foundation models, some neuromorphic hardware, and some clinical translation. Their shared name hides real differences in objective, evaluation, and risk.

This institutional growth is useful because the work requires data, hardware, experimental access, and teams that rarely live in one department. It is also dangerous if NeuroAI becomes a funding label attached after the work is designed. The field needs criteria stricter than institutional branding.

## Where the gaps remain

First, many “brain-inspired” claims stop at metaphor. A project should state the biological principle, the computational change it implies, and the test that would show the principle mattered.

Second, neural predictivity is frequently mistaken for explanation. Two models can fit the same activity using different mechanisms. Perturbation, error structure, and generalization are needed to separate them.

Third, data scale is confused with biological breadth. Millions of neural samples from a narrow device and task do not automatically produce a general model of brains.

Fourth, benchmarks reward average performance while biology often matters most in adaptation, individual difference, resource limitation, and failure.

Finally, the two directions of NeuroAI are not guaranteed to meet. Neuroscience may supply inspiration without receiving a better theory; AI may supply prediction without becoming biologically informative. The exchange succeeds only when results travel back.

NeuroAI is therefore not a new department between two old ones. It is a contract between levels of explanation. Biology must be abstracted enough to build and test. Artificial systems must be interrogated deeply enough to teach us something about minds and brains. When either side serves only as decoration, the contract has failed.

## References

Zador, A. M. (2019). A critique of pure learning and what artificial neural networks can learn from animal brains. *Nature Communications*, 10, 3770.

Richards, B. A., et al. (2019). A deep learning framework for neuroscience. *Nature Neuroscience*, 22, 1761-1770.

Macpherson, T., et al. (2021). Natural and artificial intelligence: A brief introduction to the interplay between AI and neuroscience research. *Neural Networks*, 144, 603-613.
