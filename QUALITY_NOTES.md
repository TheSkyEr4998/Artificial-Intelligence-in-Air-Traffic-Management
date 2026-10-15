# Data and result interpretation

The completed run begins with the 12 supplied monthly feature files. Deterministic month ownership removes overlapping calendar copies. It excludes invalid or nonpositive targets, recognized airport vehicle IDs, and 188 feature records whose daily ID has multiple valid Mode S addresses. The identity rule uses raw IDs and addresses, not taxi-time targets or model errors. It can remove a valid movement when a daily identifier was reused.

## Source-session support

The supplied cleaner formats epoch values as UTC clock strings without storing a timezone. The supplied snapping program parses the clock strings as naive local time. The historical host timezone is not recorded. The new source audit therefore tests UTC and Europe/London mappings instead of silently selecting one. It also applies the 2,400-second observation-gap policy found in the supplied snap program to callsigned and uncalled identifiers alike.

Out of 61,397 prepared feature rows:

- 59,683 have a unique compatible source-session interpretation and exact agreement between the inferred feature duration and snapped interval span.
- 1,714 are held for source review: 904 have unresolved label/span bounds, 858 span a raw observation gap, 46 have competing clock/session mappings, and 5 may cross a clock-offset transition. Reasons overlap, so their counts sum to more than the held-row count.

The audit keeps source clocks, labels, and predictors unchanged. It does not subtract an hour, subtract an observation gap, or choose a duration threshold. The support decision means only that the recorded route interval fits one source-observation session under the declared hypotheses. It cannot confirm an official taxi event, continuous movement, or a historical timezone.

The modeling cohort is selected with source evidence and feature-label span agreement before the date split. This deliberately narrows the target population. All performance numbers describe this screened cohort and may not transfer to every supplied feature row.

## Model result

The eight original factors are movement type, start-side hour, departing and arriving traffic counts at the movement start, cumulative angle, shortest path, gate distance, and long-edge distance. Route PCA is fitted on training records only and retained three components for 95.11% variance. PCA describes predictor variation; validation MAE selects original factors and the final model.

ANFIS with eight rules is the validation-selected model (2.6490-minute validation MAE). Its repeated December evaluation is MAE 2.8433 minutes, RMSE 7.3054 minutes, and R² 0.3138 on 5,776 records. The neural network's descriptive evaluation MAE is 2.8239 minutes. The paired whole-date bootstrap interval for ANFIS minus neural-network MAE is approximately [-0.027, 0.083] minutes and spans zero; the data do not resolve their small difference. The model scores are against inferred labels, not verified airport ground truth.

The baseline MAE is 5.1791 minutes. ANFIS and the neural network improve on this baseline on the screened, previously inspected dates. The evaluation R² values are moderate, and RMSE is much larger than MAE, showing that a smaller number of longer errors still matters. No duration-based outlier filter or prediction clipping is used.

## Evaluation status

The date split uses whole source-date groups and a one-day boundary gap. Training covers 1 January through 27 October 2021, validation covers 29 October through 10 December, and evaluation covers 12 through 31 December. These December dates were evaluated in the earlier project run. The present run is a repeated development evaluation and must not be described as a fresh independent test. The 20-date bootstrap is conditional on the frozen models and screened records; it does not include label, cohort, or training uncertainty.

This finished repository therefore documents a reproducible, source-screened retrospective benchmark with explicit evidence limits. It does not claim live taxi-start forecasting or operational suitability.
