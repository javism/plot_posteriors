#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('grayscale')
plt.rc('text', usetex=True)

# E is the test outcome
# H is the hypothesis
posterior = lambda PrEH, PrH, PrEnH: (PrEH * PrH) / (PrEH * PrH + PrEnH * (1 - PrH))


# Pr(E|H) Chance of a positive test (E) given there is a liar (H). 
# This is the chance of a true positive. Also sensitivity
PrEH = 0.7366
# Chance of a positive test (E) given that the person wasn't a lying. 
# This is the probability of a false positive, also (1-specificity)
PrEnH = 0.2445

# Example with 5% of liars
PrH = 0.05
PrHE = posterior(PrEH, PrH, PrEnH)
PrnHnE = posterior( (1-PrEnH), (1-PrH), (1-PrEH) )
print('PrHE={} - PrnHnE={}'.format(PrHE, PrnHnE))

# Let's test different hypotheses
vPrH = np.arange(0.00001, 1.0, 0.0001)
vfunc = np.vectorize(posterior)

# posterior of true positives / positive predictive value
vPrHE = vfunc(PrEH, vPrH, PrEnH)
# posterior of false positives
vPrnHE = 1 - vPrHE 

# posterior of true negatives / false predictive value
# 1-PrEnH = specificity 
# 1-PrH = prevalence of negative class
# 1-PrEH = 1-sensitivity
vPrnHnE = vfunc( (1-PrEnH), (1-vPrH), (1-PrEH) )
# posterior of false negatives
vPrHnE = 1 - vPrnHnE 

# Plot the two posteriors
fig, ax = plt.subplots()
ax.plot(vPrH, vPrHE, '-', lw=2)
ax.plot(vPrH, vPrnHnE, '-', lw=2)
plt.xlabel(r'$\displaystyle P(\textrm{Lie})$')       
plt.ylabel(r"Predictive value") 
plt.legend([r"$\displaystyle P(\textrm{Lie}\mid +)$", r"$\displaystyle P(\textrm{No-lie}\mid -)$"], 
          loc='center right')

# Anotate the point for 0.05
i = 500

ax.plot(vPrH[i], vPrHE[i], 'o', lw=2)
ax.plot(vPrH[i], vPrnHnE[i], 'o', lw=2)

ax.annotate('True positives (PPV)={:.4f} \n False positives (FDR)={:.4f}'.format(vPrHE[i],vPrnHE[i]),
            xy=(vPrH[i], vPrHE[i]),
            xycoords='data',
            xytext=(vPrH[i] + 0.07, vPrHE[i] + 0.05),
            textcoords='axes fraction',
            horizontalalignment='left',
            verticalalignment='top')

ax.annotate('True negatives (NPV)={:.4f} \n False negatives (FOR)={:.4f}'.format(vPrnHnE[i], vPrHnE[i]),
            xy=(vPrH[i], vPrnHnE[i]),
            xycoords='data',
            xytext=(vPrH[i] + 0.05, vPrnHnE[i] - 0.10),
            textcoords='axes fraction',
            horizontalalignment='left',
            verticalalignment='top')
vPrH[i]

plt.savefig('posteriors.pdf')
plt.savefig('posteriors.png')
plt.waitforbuttonpress()
