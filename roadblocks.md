# Roadblocks and Fixes

## 1. Regional bias in simulation results

After running 1,000 Monte Carlo simulations, the results showed a strong bias toward CONMEBOL teams. CONMEBOL teams had around 60% total winner probability, while UEFA had around 40%, and other confederations had almost 0%.

This looked too extreme, especially because several strong European teams ranked lower than expected.

## 2. Possible causes

The likely causes are:

* Elo ratings may be inflated for some teams because different confederations play different numbers and types of competitive matches.
* Confederation features may have added an extra regional effect.
* FIFA ranking was shown in the dashboard but was not used in the prediction model.
* Knockout matches used home/away order even when played at neutral venues.

## 3. Planned fixes

To improve the model, we are testing:

* Removing confederation features and retraining the Poisson models.
* Capping or scaling Elo difference to reduce overconfidence.
* Adding a small FIFA-rank-based correction.
* Making knockout predictions symmetric so neutral matches are not affected by bracket order.

## 4. Current strategy

The working V1 app has been saved on the main branch.

All model calibration tests are being done on a separate branch:

```text
experiment-model-calibration
```

This keeps the stable app safe while allowing new model versions to be tested cleanly.