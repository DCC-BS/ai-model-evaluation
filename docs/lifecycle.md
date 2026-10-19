# Model lifecycle

This page explains how a model comes onto the KDKP, how we replace it, and what we write down along the way. The DCC AI team runs this process. We are setting up a governance board for model decisions.

## How often we review

Every 6 to 12 months we compare our models with what is newly available. We assume that after about a year, newer open models will do the same job better. So we start looking before something breaks.

## How a switch works

1. **Candidate.** Someone proposes a new model. It has to pass the [five basic checks](selection-standard.md#the-five-basic-checks).
2. **Testing.** We run rounds 1 to 4 from [eval-layers.md](eval-layers.md): public rankings, our own tests, tests in our apps and load tests.
3. **Running both.** The old and the new model run side by side for a while. Apps and workflows get adjusted, and we compare the two in real use.
4. **Switching over.** The apps move to the new model's API URL. We first switch the old model off, and remove it completely some time later.

## Going back

Sometimes we go back to the old model. Once we switched off a new model because it wrote long chains of reasoning for tasks that needed a short answer. The public [model list](https://dcc-bs.github.io/documentation/infrastructure/models) shows its status.

## What we write down

- The status of each model (in use, switched off or removed) in the public model list.
- An internal page.
- The change history of our infrastructure.
- Load test results.
- Earlier test results, for example for speech-to-text, deep research and the comparison of decision models.
- A decision record for each model change. See [decision-record-template.md](decision-record-template.md).

Most of these records are internal. The template shows what they contain.

## What we still need to improve

When we last switched models, we didn't record every measurement we should have. The decision record now asks for all of them, round by round.
