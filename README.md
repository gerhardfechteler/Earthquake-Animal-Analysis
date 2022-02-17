# Earthquake-Animal-Analysis
 
This repository contains the code used for the analysis in the paper "Potential short-term earthquake forecasting by farm animal monitoring", published in Ethology. The paper and supplementary material are available under the following link: https://onlinelibrary.wiley.com/doi/full/10.1111/eth.13078. 

We analyse the time series of earthquake activity, measured by peak ground acceleration (PGA), and activity of various animal groups, measured by overall dynamic body acceleration (ODBA). We use fourier filtering to extract the daily activity patterns of the animals, a vector autoregressive model to extract the reactive patterns and finally a threshold model to test for Granger-type predictive patterns.

The file Functions.py contains the implemented classes and functions for the analysis, the file Code.py performs the actual analysis. The data is available upon request.
