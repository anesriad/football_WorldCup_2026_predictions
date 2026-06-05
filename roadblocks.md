# Roadblocks and Fixes

## 1. Regional bias in simulation results

After running 1,000 Monte Carlo simulations, the first version showed a strong bias toward CONMEBOL teams.

CONMEBOL teams had around 60% total winner probability, UEFA had around 40%, and other confederations had almost 0%.

This looked too extreme, especially because several strong European teams ranked lower than expected.

## 2. Possible causes

The likely causes were:

* Confederation features may have added an extra regional effect.
* Elo ratings may have been higher for some teams because different confederations play different numbers and types of competitive matches.
* FIFA ranking was shown in the dashboard but was not used directly in the prediction model.
* Knockout matches used home/away order even when played at neutral venues.

## 3. Fix applied

The main fix was to remove confederation features from the Poisson models.

The updated model now uses a cleaner feature set:

* home Elo
* away Elo
* tournament weight
* neutral venue flag

This keeps the model focused on team strength and match context, without adding a learned regional effect.

## 4. Result after the fix

After retraining the Poisson models without confederation features and rerunning 1,000 Monte Carlo simulations, the results became more coherent.

The updated total winner probabilities by confederation were roughly:

```text
UEFA:     60.7%
CONMEBOL: 32.5%
CAF:       2.6%
AFC:       2.1%
CONCACAF: 2.0%
OFC:       0.1%
```

This is more realistic than the first version because UEFA and CONMEBOL still dominate, but other confederations now have small non-zero chances.
