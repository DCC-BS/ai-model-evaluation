# Model Decision Record

We write one record for each model change. Our records are private and are kept on an internal page or in a private repository. This is the template. Copy it, fill it in and keep the result where your team keeps decisions.

## Change and motivation

Why is this change happening now? Name the model being replaced or added.

## Candidates considered

List every candidate. For each, record the result of the five [hard gates](selection-standard.md#hard-gates): GPU fit, one server, vLLM, licence, Hugging Face.

## Evaluation per layer

For each of the layers in [eval-layers.md](eval-layers.md), link to the run and list all relevant metrics, not only the headline number. Note which metrics were not measured and why.

## Load tests

Link to the load-test results before and after the change.

## Transition

When did the old and new model run side by side? What issues did we see?

## Rollback criteria

What would make us roll back? Write this before the transition starts.

## Decision

What was decided, by whom and on what date?

## Sign-off

Governance board sign-off, once a board exists. Leave empty until then.
