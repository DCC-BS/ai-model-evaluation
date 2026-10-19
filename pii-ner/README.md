# PII and NER evaluation

The code for this evaluation is not public yet. It joins this repository later.

## What we evaluate

Models that find personal data in German administrative text so that it can be anonymised: names, addresses, dates of birth, identification numbers and similar entities.

## Why

For anonymisation, a missed entity is worse than a false alarm. A false positive costs a reviewer a few seconds. A false negative can leave personal data in a published document. We therefore look at recall per entity type first and weight false negatives as the critical error.

## Status

No code, data or results here yet. We run this evaluation on our own data and will publish the method, not the data or scores.
