# Model decision record

We fill in one record for each model change and keep it on an internal page or in a private repository. Below is the empty template. Copy it, fill it in, and keep it wherever your team keeps its decisions.

## What changes and why

Why does this change happen now? Name the model you replace or add.

## Models considered

List every model you looked at. For each one, note how it did in the [five basic checks](selection-standard.md#the-five-basic-checks): fits on the GPUs, runs on one server, runs on vLLM, licence, published on Hugging Face.

## Test results per round

For each round in [eval-layers.md](eval-layers.md), link to the test run and list every measurement that matters, including the minor ones. Write down what you didn't measure, and why.

## Load tests

Link to the load test results from before and after the change.

## Running both models

When did the old and the new model run side by side? What problems came up?

## When we would go back

What would make you switch back to the old model? Write this down before both models start running side by side.

## Decision

What did you decide, who decided, and on what date?

## Approval

Approval by the governance board, once the board exists. Leave this empty until then.
