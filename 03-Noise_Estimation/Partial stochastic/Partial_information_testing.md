# Partial Information testing

We test using example 2 from the previous notes; gaussian data + gaussian noise. Our example has both missing and observed $Z$. The ratio has not been pre-set. As such, we are able to manipulate this to see the affect of increasing and decreasing the amount of data missing. 



We begin by computing a complete data solution. This allows us to check if our log likelihood is correct. In testing, each test uses a new randomly generated dataset. Rather than presenting all of the separate complete data estimates, we present a random selection.



Initially we used $\mathbb{E}[Z]$ for all missing values of $Z$ and tested for 1 set of $\sigma_{\epsilon}$ and $\sigma_z$.  Each EM was run for 100 iterations in all testing. 

$\sigma_{\epsilon} = 0.05$

$\sigma_z = 0.5$





| Ratio Z observed | Estimated $\sigma_{\epsilon}$ | Estimated $\sigma_z$ | Converged? |
| ---------------- | ----------------------------- | -------------------- | ---------- |
| 0.01             | 0.8933                        | 0.7424               | yes        |
| 0.25             | 0.7166                        | 0.6206               | yes        |
| 0.5              | 0.4890                        | 0.4872               | yes        |
| 0.75             | 0.2934                        | 0.4554               | yes        |
| 0.99             | 0.0659                        | 0.4940               | yes        |
| Complete (1)     | 0.0504                        | 0.4995               | NA         |



It is quite clear the model is learning from the observed data and not extracting any information from the expectation. After further research, we noticed that other EM models treated individual $Z_i$ as random variables rather than just $Z$ and $Z_i$ as realisations. As such we can now get a vector of $\mathbb{E}[Z_i]$ that extracts more information. 



We begin by testing the same $\sigma$ values. 

$\sigma_{\epsilon} = 0.05$

$\sigma_z = 0.5$





| Ratio Z observed | Estimated $\sigma_{\epsilon}$ | Estimated $\sigma_z$ | Converged? |
| ---------------- | ----------------------------- | -------------------- | ---------- |
| 0.01             | 0.0050                        | 0.5012               | yes        |
| 0.25             | 0.02446                       | 0.5020               | yes        |
| 0.5              | 0.0341                        | 0.5015               | yes        |
| 0.75             | 0.0434                        | 0.5018               | yes        |
| 0.99             | 0.0494                        | 0.5024               | yes        |
| Complete (1)     | 0.0496                        | 0.5027               | NA         |



Our second batch of $\sigma$

$\sigma_{\epsilon} = 0.03$

$\sigma_z = 0.7$





| Ratio Z observed | Estimated $\sigma_{\epsilon}$ | Estimated $\sigma_z$ | Converged? |
| ---------------- | ----------------------------- | -------------------- | ---------- |
| 0.01             | 0.0031                        | 0.6997               | yes        |
| 0.25             | 0.0148                        | 0.7015               | yes        |
| 0.5              | 0.0212                        | 0.7060               | yes        |
| 0.75             | 0.0261                        | 0.7009               | yes        |
| 0.99             | 0.0296                        | 0.7042               | yes        |
| Complete (1)     | 0.0298                        | 0.7041               | NA         |



Our final batch:

$\sigma_{\epsilon} = 0.09$

$\sigma_z = 0.4$





| Ratio Z observed | Estimated $\sigma_{\epsilon}$ | Estimated $\sigma_z$ | Converged? |
| ---------------- | ----------------------------- | -------------------- | ---------- |
| 0.01             | 0.0078                        | 0.4103               | yes        |
| 0.25             | 0.0443                        | 0.4088               | yes        |
| 0.5              | 0.0642                        | 0.4060               | yes        |
| 0.75             | 0.0776                        | 0.4110               | yes        |
| 0.99             | 0.0902                        | 0.4006               | yes        |
| Complete (1)     | 0.0905                        | 0.4004               | NA         |





* It is quite clear from the results that making the change to expectation has improved our estimation of $\sigma_z$ drastically, even with extremely little observed values of $Z$. This is likely due to our new expected values of $Z$ maintaining the variability in the underlying data whereas before we had fixed value. 
* The EM is still unable to get a good estimation of $\sigma_e$. It is very clear that increasing the ratio of observed data results in a significant improvement in $\sigma_{\epsilon}$ estimation. 



To check if the second observation above is due to $\sigma_{\epsilon}$ estimation or because it is the smaller value, we conduct another test. This time making $\sigma_{\epsilon}$ the larger value. Since technically we are only summing two gaussians, which one we treat as the noise is arbitrary.



Reversing size of $\sigma$

$\sigma_{\epsilon} = 0.5$

$\sigma_z = 0.05$





| Ratio Z observed | Estimated $\sigma_{\epsilon}$ | Estimated $\sigma_z$ | Converged? |
| ---------------- | ----------------------------- | -------------------- | ---------- |
| 0.01             | 0.0506                        | 0.5047               | yes        |
| 0.25             | 0.2528                        | 0.4779               | yes        |
| 0.5              | 0.3511                        | 0.4302               | yes        |
| 0.75             | 0.4312                        | 0.3278               | yes        |
| 0.99             | 0.4996                        | 0.0842               | yes        |
| Complete (1)     | 0.5017                        | 0.0506               | NA         |





* It is quite evident that increasing the amount of observed information had a significant impact on the estimated values. 
* The estimates for the ratio 0.01 is peculiar. These values reversed would be a very good estimate.  This could be a coincidence. To ensure we check for another pair of values.



 

Reversing size of $\sigma$, checking for abnormal behaviour at ratio = 0.01

$\sigma_{\epsilon} = 0.7$

$\sigma_z = 0.03$





| Ratio Z observed | Estimated $\sigma_{\epsilon}$ | Estimated $\sigma_z$ | Converged? |
| ---------------- | ----------------------------- | -------------------- | ---------- |
| 0.01             | 0.0709                        | 0.6890               | yes        |

* It is clear that the EM did not estimate the $\sigma$ values in reverse order here.  The value of $\sigma_{\epsilon}$ on the previous table may just be coincidental. 
* One may note, the estimate of $\sigma_z$ is extremely to close to the true value of $\sigma_{\epsilon}$. It is quite likely that when there is insufficient observed data $\sigma_z$ simply estimates the larger $\sigma$.   





### Thoughts

Usually when using EM for missing data, we are predicting model parameters for some f(X) = Y. Since this is usually deterministic, it is much easier to estimate these than the stochastic X in our case. 