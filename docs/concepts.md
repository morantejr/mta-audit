# Concepts

## Events and journeys

Input is one row per observed event. Required semantic fields are user identity,
timestamp, channel, and conversion indicator. `ColumnMapping` maps source names
without mutating the input. A journey is the ordered touch sequence ending at a
conversion or, for non-converters, the final observed event.

Lookback windows bound available history. Conversion windows test a stricter
assumption about how old a touch may be when it receives credit.

## Attribution and reliability

Attribution allocates observed conversion value; reliability describes how much
the allocation changes under plausible analytical and tracking choices. A model
can be internally valid while producing an unreliable business result.

The report separates findings, check scores, six reliability components, and the
weighted overall score. Missing components remain visibly absent.

The public Criteo dataset is supported through `mta_audit.datasets` so
dataset-specific conversion reconstruction never leaks into the core engine.

## Simulations

Corruption simulations preserve the caller's frame and use deterministic random
sampling by default. They fragment identities, remove eligible non-conversion
touches, or duplicate/noise events, then rebuild journeys and compare channel
shares with the baseline.

> Attribution is observational and does not establish incrementality or causal impact.
