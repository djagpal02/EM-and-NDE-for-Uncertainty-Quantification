# Partial Information

We set up a problem where we have some of the underlying data available. The overall problem is still:
$$
Y = Z + \epsilon
$$
Let
$$
\epsilon \sim N(0,\sigma_{\epsilon}^2)
$$
To maintain generality for as long as possible we will not yet define a distribution for $Z$. For know let us assume it is known.



Let the dataset be of size $S$ with $N$ observed and $M$ missing datapoints. ($S = N + M$)



Let 
$$
\theta = \{\theta_Z, \sigma_{\epsilon}\}
$$
where $\theta_Z$ is the set of parameters required to define the known distribution of $Z$.





## Complete LL

We begin by attempting to compute the log likelihood as if the data were complete.
$$
\begin{eqnarray}
ll(\theta|Y, Z) & = & logP(Y,Z|\theta) \\

				& = & log[P(Y|Z,\theta)P(Z|\theta)] \\
				
				& = & log\left[\prod_{n=0}^N P(y_n|z_n,\theta)P(z_n|\theta)\ \prod_{m=0}^M P(y_m|z_m,\theta)P(z_m|\theta) \right] \\
				
				& = & \sum_{n=0}^N log[P(y_n|z_n,\theta)P(z_n|\theta)] + \sum_{m=0}^M log[P(y_m|z_m,\theta)P(z_m|\theta)]
\end{eqnarray}
$$
It is clear that both the parts with data observed and data missing are contributing in a similar way. Of course since we do not actually have data for $z_m$ we must take expectations.



## Example 1: Underlying Bernoulli Z

For this example we assume 
$$
Z \sim Bern(p)
$$
thus,
$$
\theta = \{p,\sigma_{\epsilon}\}
$$


We can now input distribution information into the log likelihood formula above.
$$
\begin{eqnarray}
ll(\theta|Y, Z) & = & \sum_{n=0}^N log[P(y_n|z_n,\theta)P(z_n|\theta)] + \sum_{m=0}^M log[P(y_m|z_m,\theta)P(z_m|\theta)] \\

				& = & \sum_{n=0}^N log \left[ \dfrac{1}{\sqrt{2\pi}\ \sigma_{\epsilon}} e^{-\dfrac{(y_n-z_n)^2}{2\sigma_{\epsilon}^2}} p^z_n(1-p)^{1-z_n)}\right] + \sum_{m=0}^M log \left[ \dfrac{1}{\sqrt{2\pi}\ \sigma_{\epsilon}} e^{-\dfrac{(y_m-z_m)^2}{2\sigma_{\epsilon}^2}} p^z_m(1-p)^{1-z_m)}\right]\\

				& = & \sum_{n=0}^N \left[ -log(\sqrt{2\pi}) - log(\sigma_{\epsilon}) - \dfrac{(y_n - z_n)^2}{2\sigma_{\epsilon}^2} + z_n log(p) + (1-z_n)log(1-p)\right] + \sum_{m=0}^M \left[ -log(\sqrt{2\pi}) - log(\sigma_{\epsilon}) - \dfrac{(y_m - z_m)^2}{2\sigma_{\epsilon}^2} + z_m log(p) + (1-z_m)log(1-p)\right] \\
				
				& = & -Nlog(\sqrt{2\pi}) - Nlog(\sigma_{\epsilon}) - Mlog(\sqrt{2\pi}) - Mlog(\sigma_{\epsilon}) + \sum_{n=0}^N \left[ - \dfrac{(y_n - z_n)^2}{2\sigma_{\epsilon}^2} + z_n log(p) + (1-z_n)log(1-p)\right] + \sum_{m=0}^M \left[  - \dfrac{(y_m - z_m)^2}{2\sigma_{\epsilon}^2} + z_m log(p) + (1-z_m)log(1-p)\right] \\
				
				& = & -Slog(\sqrt{2\pi}) - Slog(\sigma_{\epsilon}) - \dfrac{\sum_{n=0}^N(y_n - z_n)^2}{2\sigma_{\epsilon}^2} - \dfrac{\sum_{m=0}^M(y_m - z_m)^2}{2\sigma_{\epsilon}^2} + log(p) \left(\sum_{n=0}^N z_n  + \sum_{m=0}^M z_m\right) + log(1-p) \left(\sum_{n=0}^N (1 - z_n)  + \sum_{m=0}^M (1 - z_m) \right) \\
				
				& = &  -Slog(\sqrt{2\pi}) - Slog(\sigma_{\epsilon}) -\dfrac{\sum_{n=0}^N y_n^2}{2\sigma_{\epsilon}^2} + \dfrac{\sum_{n=0}^N y_nz_n}{\sigma_{\epsilon}^2} - \dfrac{\sum_{n=0}^N z_n^2}{2\sigma_{\epsilon}^2}  -\dfrac{\sum_{m=0}^M y_m^2}{2\sigma_{\epsilon}^2} + \dfrac{\sum_{m=0}^M y_mz_m}{\sigma_{\epsilon}^2} - \dfrac{\sum_{m=0}^M z_m^2}{2\sigma_{\epsilon}^2}  + log(p) \left(\sum_{n=0}^N z_n  + \sum_{m=0}^M z_m\right) + log(1-p) \left(\sum_{n=0}^N (1 - z_n)  + \sum_{m=0}^M (1 - z_m) \right)
\end{eqnarray}
$$




### Prior Z

Since we do not have all the $M$ datapoints we must take an expectation using the prior of  $Z$ given the previous parameter values. We must first compute this prior.

$$
\begin{eqnarray}
P(Z_M|Y_M,Z_N,\theta_{old}) & \propto & P(Y_M|Z_M,Z_N,\theta_{old})P(Z_m|\theta_{old})\\
							
							& = & P(Y_M|Z_M,\theta_{old})P(Z_m|\theta_{old}) \\
							
							& = & \dfrac{1}{\sqrt{2\pi}\sigma_{\epsilon}^2}e^{-\dfrac{(Y_M-Z_M)^2}{2\sigma_{\epsilon}^2}}p^{Z_M}(1-p)^{Z_M}
\end{eqnarray}
$$


### Expectation

Due to us using a Bernoulli distribution, 
$$
\begin{eqnarray}
\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M] & = &  P(Z_M = 1|Y_M,Z_N,\theta_{old}) \\

											& \propto & \dfrac{1}{\sqrt{2\pi}\sigma_{\epsilon}^2}e^{-\dfrac{(Y_M-1)^2}{2\sigma_{\epsilon}^2}}p
\end{eqnarray}
$$
Similarly, 
$$
\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2] = \mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M]
$$




### Auxiliary Function

We can now substitute this back into our ll to get our auxiliary function.
$$
\begin{eqnarray}
Q(\theta|\theta_{old}) & = &  -Slog(\sqrt{2\pi}) - Slog(\sigma_{\epsilon}) -\dfrac{\sum_{n=0}^N y_n^2}{2\sigma_{\epsilon}^2} + \dfrac{\sum_{n=0}^N y_nz_n}{\sigma_{\epsilon}^2} - \dfrac{\sum_{n=0}^N z_n^2}{2\sigma_{\epsilon}^2}  -\dfrac{\sum_{m=0}^M y_m^2}{2\sigma_{\epsilon}^2} + \dfrac{\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M]\sum_{m=0}^M y_m}{\sigma_{\epsilon}^2} - \dfrac{M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2]}{2\sigma_{\epsilon}^2}  + log(p) \left(\sum_{n=0}^N z_n  + M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M]\right) + log(1-p) \left(\sum_{n=0}^N (1 - z_n)  + M(1 - \mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M]) \right) \\

\end{eqnarray}
$$


### Update parameters - p

$$
\begin{eqnarray}
\dfrac{\partial Q}{\partial p} & = & \dfrac{\sum_{n=0}^N z_n  + M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M]}{p} + \dfrac{\sum_{n=0}^N (1 - z_n)  + M(1 - \mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M]) }{p-1} = 0
\end{eqnarray}
$$


$$
\begin{eqnarray}
0 & = & \dfrac{\sum_{n=0}^N z_n  + M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M]}{p} + \dfrac{\sum_{n=0}^N (1 - z_n)  + M(1 - \mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M]) }{p-1} \\

 & = & p\sum_{n=0}^N z_n - \sum_{n=0}^N z_n + pM\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M] - M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M] + p\sum_{n=0}^N (1 - z_n) + pM - pM\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M] \\

 & = & p\sum_{n=0}^N z_n - \sum_{n=0}^N z_n - M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M] + p\sum_{n=0}^N (1 - z_n) + pM \\

 & = & p\sum_{n=0}^N z_n - \sum_{n=0}^N z_n - M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M] + pN - p\sum_{n=0}^N z_n + pM \\
 
 & = & - \sum_{n=0}^N z_n - M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M] + pN + pM \\

p(N+M) & = & \sum_{n=0}^N z_n + M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M] \\

p & = & \dfrac{\sum_{n=0}^N z_n + M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M]}{S}
\end{eqnarray}
$$


### Update parameter - $\sigma_{\epsilon}$ #

$$
\dfrac{\partial Q}{\partial \sigma_{\epsilon}} = -\dfrac{S}{\sigma_{\epsilon}} - \dfrac{-\sum_{n=0}^N y_n^2 + 2\sum_{n=0}^N y_nz_n - \sum_{n=0}^N z_n^2 - \sum_{m=0}^M y_m^2 + 2\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M]\sum_{m=0}^M y_m - M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2]}{\sigma_{\epsilon}^3}  \\
$$

$$
\begin{eqnarray}
0 & = & -\dfrac{S}{\sigma_{\epsilon}} - \dfrac{-\sum_{n=0}^N y_n^2 + 2\sum_{n=0}^N y_nz_n - \sum_{n=0}^N z_n^2 - \sum_{m=0}^M y_m^2 + 2\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M]\sum_{m=0}^M y_m - M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2]}{\sigma_{\epsilon}^3}  \\

S \sigma_{\epsilon}^2  & = & \sum_{n=0}^N y_n^2 - 2\sum_{n=0}^N y_nz_n + \sum_{n=0}^N z_n^2 + \sum_{m=0}^M y_m^2 - 2\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M]\sum_{m=0}^M y_m + M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2] \\

					& = & \sum_{n=0}^N (y_n - z_n)^2 + \sum_{m=0}^M y_m^2 - 2\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M]\sum_{m=0}^M y_m + M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2] \\
					
					& = & \sum_{n=0}^N (y_n - z_n)^2 + \sum_{m=0}^M \left[ y_m^2 - 2\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M] y_m + \mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2] \right] \\
					
\sigma_{\epsilon} & = & \sqrt{\dfrac{\sum_{n=0}^N (y_n - z_n)^2 + \sum_{m=0}^M \left[ y_m^2 - 2\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M] y_m + \mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2] \right]}{S}}
\end{eqnarray}
$$

## Example 2: Underlying Normal Z

For this example we assume 
$$
Z \sim N(0,\sigma_{z}^2)
$$
thus,
$$
\theta = \{\sigma_z,\sigma_{\epsilon}\}
$$


We can now input distribution information into the log likelihood formula above.
$$
\begin{eqnarray}
ll(\theta|Y, Z) & = & \sum_{n=0}^N log[P(y_n|z_n,\theta)P(z_n|\theta)] + \sum_{m=0}^M log[P(y_m|z_m,\theta)P(z_m|\theta)] \\

				& = & \sum_{n=0}^N log \left[ \dfrac{1}{\sqrt{2\pi}\ \sigma_{\epsilon}} e^{-\dfrac{(y_n-z_n)^2}{2\sigma_{\epsilon}^2}}  \dfrac{1}{\sqrt{2\pi}\ \sigma_{z}} e^{-\dfrac{z_n^2}{2\sigma_{z}^2}}\right] + \sum_{m=0}^M log \left[ \dfrac{1}{\sqrt{2\pi}\ \sigma_{\epsilon}} e^{-\dfrac{(y_m-z_m)^2}{2\sigma_{\epsilon}^2}}\dfrac{1}{\sqrt{2\pi}\ \sigma_{z}} e^{-\dfrac{z_m^2}{2\sigma_{z}^2}}\right]\\ 
				
				& = & \sum_{n=0}^N log \left[ \dfrac{1}{2\pi\ \sigma_{\epsilon}\sigma_z} e^{-\dfrac{(y_n-z_n)^2}{2\sigma_{\epsilon}^2} -\dfrac{z_n^2}{2\sigma_{z}^2}}  \right] + \sum_{m=0}^M log \left[ \dfrac{1}{2\pi\ \sigma_{\epsilon}\sigma_z} e^{-\dfrac{(y_m-z_m)^2}{2\sigma_{\epsilon}^2} -\dfrac{z_m^2}{2\sigma_{z}^2}}  \right] \\
				
				& = & \sum_{n=0}^N \left[ -log(2\pi)  - log(\sigma_{\epsilon}) - log(\sigma_z) -\dfrac{(y_n-z_n)^2}{2\sigma_{\epsilon}^2} -\dfrac{z_n^2}{2\sigma_{z}^2} \right] + \sum_{m=0}^M \left[ -log(2\pi)  - log(\sigma_{\epsilon}) - log(\sigma_z) -\dfrac{(y_m-z_m)^2}{2\sigma_{\epsilon}^2} -\dfrac{z_m^2}{2\sigma_{z}^2} \right] \\
				
				& = & -Nlog(2\pi)  - Nlog(\sigma_{\epsilon}) - Nlog(\sigma_z) -\dfrac{\sum_{n=0}^N(y_n-z_n)^2}{2\sigma_{\epsilon}^2} -\dfrac{\sum_{n=0}^Nz_n^2}{2\sigma_{z}^2} -Mlog(2\pi)  - Mlog(\sigma_{\epsilon}) - Mlog(\sigma_z) -\dfrac{\sum_{m=0}^M(y_m-z_m)^2}{2\sigma_{\epsilon}^2} -\dfrac{\sum_{m=0}^Mz_m^2}{2\sigma_{z}^2} \\
				
				& = & -Slog(2\pi)  - Slog(\sigma_{\epsilon}) - Slog(\sigma_z) -\dfrac{\sum_{n=0}^N(y_n-z_n)^2}{2\sigma_{\epsilon}^2} -\dfrac{\sum_{n=0}^Nz_n^2}{2\sigma_{z}^2} -\dfrac{\sum_{m=0}^M(y_m-z_m)^2}{2\sigma_{\epsilon}^2} -\dfrac{\sum_{m=0}^Mz_m^2}{2\sigma_{z}^2} \\
				
				& = & -Slog(2\pi)  - Slog(\sigma_{\epsilon}) - Slog(\sigma_z) -\dfrac{\sum_{n=0}^N(y_n-z_n)^2 + \sum_{m=0}^M(y_m-z_m)^2}{2\sigma_{\epsilon}^2} -\dfrac{\sum_{n=0}^Nz_n^2 + \sum_{m=0}^Mz_m^2}{2\sigma_{z}^2}

				
\end{eqnarray}
$$


### Prior Z

Since we do not have all the $M$ datapoints we must take an expectation using the prior of  $Z$ given the previous parameter values. We must first compute this prior.
$$
\begin{eqnarray}
P(Z_M|Y_M,Z_N,\theta_{old}) & \propto & P(Y_M|Z_M,Z_N,\theta_{old})P(Z_m|\theta_{old})\\
							
							& = & P(Y_M|Z_M,\theta_{old})P(Z_m|\theta_{old}) \\
							
							& \propto & e^{-\dfrac{(Y_M-Z_M)^2}{2\sigma_{\epsilon}^2}}e^{-\dfrac{Z_M^2}{2\sigma_{z}^2}} \\
							
							& = & e^{-\dfrac{1}{2}\left[Y_m^2\sigma_{\epsilon}^{-2} -2Y_mZ_m\sigma_{\epsilon}^{-2} + Z_m^2\sigma_{\epsilon}^{-2} + Z_m^2\sigma_{z}^{-2}\right]}
\end{eqnarray}
$$
We break down the exponent term ignoring the $-\dfrac{1}{2}$.
$$
Y_m^2\sigma_{\epsilon}^{-2} -2Y_mZ_m\sigma_{\epsilon}^{-2} + Z_m^2\sigma_{\epsilon}^{-2} + Z_m^2\sigma_{z}^{-2} =(\sigma_{\epsilon}^{-2}+\sigma_{z}^{-2})Z_m^2 - 2Y_m\sigma_{\epsilon}^{-2}Z_m  + Y_m^2\sigma_{\epsilon}^{-2}
$$


Let:
$$
A = \sigma_{\epsilon}^{-2}+\sigma_{z}^{-2}
$$

$$
B = Y_m\sigma_{\epsilon}^{-2}
$$

$$
C = Y_m^2 \sigma_{\epsilon}^{-2}
$$

Note that this is a quadratic in $Z_m$. As such we can complete the square:
$$
\begin{eqnarray}
(\sigma_{\epsilon}^{-2}+\sigma_{z}^{-2})Z_m^2 - 2Y_m\sigma_{\epsilon}^{-2}Z_m  + Y_m^2\sigma_{\epsilon}^{-2} & = & AZ_m^2 - 2BZ_m + C  \\

			& = & A( Z_m^2 - \dfrac{B}{A})^2 -\dfrac{B^2}{A^2} + C
\end{eqnarray}
$$
Thus,
$$
\begin{eqnarray}
P(Z_M|Y_M,Z_N,\theta_{old}) & \propto &  e^{-\dfrac{1}{2}\left[Y_m^2\sigma_{\epsilon}^{-2} -2Y_mZ_m\sigma_{\epsilon}^{-2} + Z_m^2\sigma_{\epsilon}^{-2} + Z_m^2\sigma_{z}^{-2}\right]} \\

							& = & e^{-\dfrac{1}{2}A\left( Z_m^2 - \dfrac{B}{A}\right)^2 -\dfrac{B^2}{A^2} + C} \\
							
							& \propto & e^{-\dfrac{1}{2}A\left( Z_m^2 - \dfrac{B}{A}\right)^2 }
\end{eqnarray}
$$
This is clearly a scaled gaussian with 
$$
\mu = \dfrac{B}{A}
$$

$$
\sigma = A^{-1}
$$

### Expectation

Since $P(Z_M|Y_M,Z_N,\theta_{old})$ is a normal distribution, 
$$
\begin{eqnarray}
\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M] & = &  \mu \\

											& = & \dfrac{B}{A}
\end{eqnarray}
$$
Using $Var[X] = E[X^2] - E[X]^2$
$$
\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2] = A^{-1} + \dfrac{B^2}{A^2}
$$

### Auxiliary Function

We can now substitute this back into our ll to get our auxiliary function.
$$
\begin{eqnarray}
Q(\theta|\theta_{old}) & = &  -Slog(2\pi)  - Slog(\sigma_{\epsilon}) - Slog(\sigma_z) -\dfrac{\sum_{n=0}^N(y_n-z_n)^2 + \sum_{m=0}^M(y_m-z_m)^2}{2\sigma_{\epsilon}^2} -\dfrac{\sum_{n=0}^Nz_n^2 + \sum_{m=0}^Mz_m^2}{2\sigma_{z}^2} \\

					  & = & -Slog(2\pi)  - Slog(\sigma_{\epsilon}) - Slog(\sigma_z) -\dfrac{\sum_{n=0}^N(y_n-z_n)^2 + \sum_{m=0}^My_m^2 -2\sum_{m=0}^My_mz_m + \sum_{m=0}^M z_m^2}{2\sigma_{\epsilon}^2} -\dfrac{\sum_{n=0}^Nz_n^2 + \sum_{m=0}^Mz_m^2}{2\sigma_{z}^2} \\
					  
					  & = & -Slog(2\pi)  - Slog(\sigma_{\epsilon}) - Slog(\sigma_z) -\dfrac{\sum_{n=0}^N(y_n-z_n)^2 + \sum_{m=0}^My_m^2 -2\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M]\sum_{m=0}^My_m + M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2]}{2\sigma_{\epsilon}^2} -\dfrac{\sum_{n=0}^Nz_n^2 + M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2]}{2\sigma_{z}^2} \\

\end{eqnarray}
$$

### 





### Update Parameter - $\sigma_{\epsilon}$

$$
\begin{eqnarray}
\dfrac{\partial Q}{\partial \sigma_{\epsilon}} & = & - \dfrac{S}{\sigma_{\epsilon}} + \dfrac{\sum_{n=0}^N(y_n-z_n)^2 + \sum_{m=0}^My_m^2 -2\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M]\sum_{m=0}^My_m + M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2]}{\sigma_{\epsilon}^3}
\end{eqnarray}
$$

$$
\begin{eqnarray}
0 & = & - \dfrac{S}{\sigma_{\epsilon}} + \dfrac{\sum_{n=0}^N(y_n-z_n)^2 + \sum_{m=0}^My_m^2 -2\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M]\sum_{m=0}^My_m + M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2]}{\sigma_{\epsilon}^3} \\

S\sigma_{\epsilon}^2 	& = & \sum_{n=0}^N(y_n-z_n)^2 + \sum_{m=0}^My_m^2 -2\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M]\sum_{m=0}^My_m + M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2] \\

\sigma_{\epsilon}^2 	& = & \dfrac{\sum_{n=0}^N(y_n-z_n)^2 + \sum_{m=0}^My_m^2 -2\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M]\sum_{m=0}^My_m + M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2]}{S} \\

\sigma_{\epsilon}		& = & \sqrt{\dfrac{\sum_{n=0}^N(y_n-z_n)^2 + \sum_{m=0}^My_m^2 -2\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M]\sum_{m=0}^My_m + M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2]}{S}}
\end{eqnarray}
$$



### Update Parameter - $\sigma_z$

$$
\dfrac{\partial Q}{\partial \sigma_z} = -\dfrac{S}{\sigma_z} +\dfrac{\sum_{n=0}^Nz_n^2 + M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2]}{\sigma_{z}^3}
$$

$$
\begin{eqnarray}
0 & = & -\dfrac{S}{\sigma_z} +\dfrac{\sum_{n=0}^Nz_n^2 + M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2]}{\sigma_{z}^3} \\

S\sigma_z^2 	& = & \sum_{n=0}^Nz_n^2 + M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2] \\

\sigma_z  & = & \sqrt{\dfrac{\sum_{n=0}^Nz_n^2 + M\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2]}{S}}
\end{eqnarray}
$$




