# Model lifecycle

This page describes how a model enters the KDKP, how it is replaced and how we record the decision. The DCC AI team runs this process. A governance board for model decisions is being established.

## Cadence

We review models every 6 to 12 months. We assume a model that is a year old has been overtaken by newer open models, so we do not wait for a failure to look at alternatives.

## Steps

1. **Candidate.** A new model is proposed. It must pass the [hard gates](selection-standard.md#hard-gates).
2. **Evaluation.** We run layers 1 to 4 from [eval-layers.md](eval-layers.md): public benchmarks, our own benchmarks, app and API evals, load tests.
3. **Transition phase.** The old and the new model run side by side. Apps and processes adapt. We compare the two and collect what we see in real use.
4. **Cutover.** Apps switch to the new API URL. The old model is deactivated first and retired later.

## Rollback

Rollback is a real outcome of the transition phase. In one case a model was deactivated because its reasoning traces were far longer than the tasks needed. The status is recorded in the public [model catalog](https://dcc-bs.github.io/documentation/infrastructure/models).

## Records

We keep these records:

- Active, deactivated and retired status in the public model catalog.
- An internal page.
- The infrastructure change history.
- Load-test results.
- Evaluation result history, for example for speech-to-text, deep research and the decision-model comparison.
- A Model Decision Record for each model change. See [decision-record-template.md](decision-record-template.md).

Most of these are private. We publish that they exist and what they contain, not their contents.

## Gaps

At the last model switch we did not capture every relevant metric. The Model Decision Record asks for all relevant metrics per layer so that this does not repeat.
