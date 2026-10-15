# Interview notes

## One-minute explanation

“I built a reproducible Manchester Airport taxi-time modelling pipeline from twelve monthly feature files. It audits the feature records against cleaned observations and map-snapped intervals, screens records whose inferred journey duration lacks unique source-session support, and compares a polynomial model, a neural network, and a trainable ANFIS. The validation-selected ANFIS achieved a 2.84-minute MAE on a 5,776-record screened cohort. Those December dates had already been inspected in earlier development, and the labels remain inferred; I present this as a retrospective benchmark, not a fresh operational forecast.”

## Design choices

- **Monthly data preparation:** schema checks, source-month tracking, deterministic calendar ownership, valid target/type checks, vehicle-ID rules, and explicit aircraft-address ambiguity audit.
- **Source-session audit:** compares UTC and Europe/London clock interpretations because the historical timezone is missing. It applies a source-derived 40-minute gap policy independently of callsign presence and preserves reasons for every held record.
- **Chronology:** split on complete source-date groups into training, validation, and evaluation windows with a one-day boundary gap. Dates are a proxy because exact taxi start/end timestamps are unavailable.
- **Feature selection:** fit route PCA on training only; keep context factors by design; shortlist original route factors and select one 7/8-factor set using a fixed validation probe. This avoids claiming that PCA component weights measure target importance.
- **Model comparison:** fit polynomial Ridge, MLP and first-order Sugeno ANFIS with the same selected inputs and training rows. Validation chooses the model; test metrics never change the winner.
- **ANFIS:** cluster Gaussian membership functions, optimize premise centres/widths with Adam, and solve affine consequents using regularized least squares. It is implemented in Python; MATLAB is not required.
- **Prediction:** saved artifacts use a common input encoder and return minutes without clipping. Imputation and marginal training-range flags are diagnostics, not confidence intervals.

## Results to describe carefully

ANFIS was selected with validation MAE 2.6490 minutes. Its repeated evaluation scores are MAE 2.8433, RMSE 7.3054 and R² 0.3138. The neural network's repeated MAE is 2.8239, but the paired whole-date interval for the ANFIS-minus-neural MAE difference spans zero. Both improve on the 5.1791-minute median baseline for this screened cohort. The test dates were previously inspected, and the target labels are inferred from the supplied feature pipeline.

The retained 59,683 records are those supported by a unique source-observation session under declared timestamp hypotheses. This is a transparent eligibility screen, not official event validation. Route and endpoint-derived predictors are available after the journey, so the prediction task is retrospective.
