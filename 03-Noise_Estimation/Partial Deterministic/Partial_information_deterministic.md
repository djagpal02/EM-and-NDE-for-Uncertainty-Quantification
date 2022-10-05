# Deterministic Partial Information

We set up a problem where we have some of the underlying data available. The overall problem is still:
$$
Y = f(Z) + \epsilon
$$
Let
$$
\epsilon \sim N(0,\sigma^2)
$$
$$
Z \sim N(0,1)
$$

$$
f(Z) = W Z
$$



Let the dataset be of size $S$ with $N$ observed and $M$ missing datapoints. ($S = N + M$). 

Let $Z_N$ and $Z_M$ correspond to the set of observed and missing data respectively. Let $Y_N$ and $Y_M$ be $Y$ values corresponding to their respective sets of $Z$.



Let 
$$
\theta = \{W, \sigma\}
$$
where $\theta_Z$ is the set of parameters required to define the known distribution of $Z$.







## Complete LL

We begin by attempting to compute the log likelihood as if the data were complete.
$$
\begin{eqnarray}
ll(\theta|Y, Z) & = & logP(Y,Z|\theta) \\

				& = & log[P(Y|Z,\theta)P(Z|\theta)] \\
				
				& = & log\left[\prod_{n=0}^N P(y_n|z_n,\theta)\ \prod_{m=0}^M P(y_m|z_m,\theta)\ \prod_{m=0}^MP(z_m|\theta) \right] \\
				
				& = & \sum_{n=0}^N log[P(y_n|z_n,\theta)] + \sum_{m=0}^M log[P(y_m|z_m,\theta)] + \sum_{m=0}^M log[P(z_m|\theta)]
\end{eqnarray}
$$
It is clear that both the parts with data observed and data missing are contributing in a similar way. Of course since we do not actually have data for $z_m$ we must take expectations. However we do have the prior of $Z_M$, thus we know that its value will be constant (it does not depend on theta). As such we can replace it with a constant.


$$
\begin{eqnarray}

ll(\theta|Y, Z)  & = & \sum_{n=0}^N log[P(y_n|z_n,\theta)] + \sum_{m=0}^M log[P(y_m|z_m,\theta)] + \sum_{m=0}^M log[P(z_m|\theta)] \\

				& = & \sum_{n=0}^N log[P(y_n|z_n,\theta)] + \sum_{m=0}^M log[P(y_m|z_m,\theta)] + \sum_{m=0}^M log[P(z_m)] \\
				
				& = & \sum_{n=0}^N log[P(y_n|z_n,\theta)] + \sum_{m=0}^M log[P(y_m|z_m,\theta)] + K


\end{eqnarray}
$$
for some constant K.
$$
\begin{eqnarray}
ll(\theta|Y,Z) & = &  \sum_{n=0}^N log[P(y_n|z_n,\theta)] + \sum_{m=0}^M log[P(y_m|z_m,\theta)] + K \\

			   & = & \sum_{n=0}^N \left[-log(\sqrt{2\pi})-log(\sigma) - \dfrac{(y_n - Wz_n)^2}{2\sigma^2}\right] + \sum_{m=0}^M \left[-log(\sqrt{2\pi})-log(\sigma) - \dfrac{(y_m - Wz_m)^2}{2\sigma^2}\right] + K \\
			   
\end{eqnarray}
$$


### Prior Z

Since we do not have all the $M$ datapoints we must take an expectation using the prior of  $Z$ given the previous parameter values. We must first compute this prior.
$$
\begin{eqnarray}
P(Z_M|Y,Z_N,\theta_{old}) & \propto & P(Y_M|Z_M,Z_N,\theta_{old})P(Z_m|\theta_{old})\\
							
							& = & P(Y_M|Z_M,\theta_{old})P(Z_m|\theta_{old}) \\
							
							& \propto & e^{-\dfrac{(Y_M-WZ_M)^2}{2\sigma^2}}e^{-\dfrac{Z_M^2}{2}} \\
							
							& = & e^{-\dfrac{1}{2}\left[Y_M^2\sigma^{-2} -2WY_MZ_M\sigma^{-2} + W^2Z_M^2\sigma^{-2} + Z_M^2\right]}
\end{eqnarray}
$$
We break down the exponent term ignoring the $-\dfrac{1}{2}$.
$$
Y_M^2\sigma^{-2} -2WY_MZ_M\sigma^{-2} + W^2Z_M^2\sigma^{-2} + Z_M^2 =(W^2\sigma^{-2}+1)Z_M^2 - 2WY_M\sigma^{-2}Z_M  + Y_M^2\sigma^{-2}
$$


Let:
$$
A = W^2\sigma^{-2}+1
$$

$$
B = WY_M\sigma^{-2}
$$

$$
C = Y_M^2 \sigma^{-2}
$$

Note that this is a quadratic in $Z_m$. As such we can complete the square:
$$
\begin{eqnarray}
(W^2\sigma^{-2}+1)Z_m^2 - 2WY_m\sigma^{-2}Z_m  + Y_m^2\sigma^{-2}& = & AZ_M^2 - 2BZ_M + C  \\

			& = & A( Z_M^2 - \dfrac{B}{A})^2 -\dfrac{B^2}{A^2} + C
\end{eqnarray}
$$
Thus,
$$
\begin{eqnarray}
P(Z_M|Y_M,Z_N,\theta_{old}) & \propto &  e^{-\dfrac{1}{2}\left[(W^2\sigma^{-2}+1)Z_m^2 - 2WY_m\sigma^{-2}Z_m  + Y_m^2\sigma^{-2}\right]} \\

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

We can substitute these expectations into our ll to obtain our auxiliary function.
$$
\begin{eqnarray}
Q(\theta|\theta_{old})  & = & \sum_{n=0}^N \left[-log(\sqrt{2\pi})-log(\sigma) - \dfrac{(y_n - Wz_n)^2}{2\sigma^2}\right] + \sum_{m=0}^M \left[-log(\sqrt{2\pi})-log(\sigma) - \dfrac{(y_m - Wz_m)^2}{2\sigma^2}\right] + K \\

				& = & -Slog(\sqrt{2\pi}) - Slog(\sigma) + \sum_{n=0}^N \left[- \dfrac{(y_n - Wz_n)^2}{2\sigma^2}\right] + \sum_{m=0}^M \left[- \dfrac{(y_m - Wz_m)^2}{2\sigma^2}\right] + K \\
				
				& = & -Slog(\sqrt{2\pi}) - Slog(\sigma)  - \dfrac{\sum_{n=0}^N(y_n - Wz_n)^2 + \sum_{m=0}^M (y_m - Wz_m)^2}{2\sigma^2} + K \\
				
				& = & -Slog(\sqrt{2\pi}) - Slog(\sigma) - \dfrac{\sum_{n=0}^N(y_n - Wz_n)^2 + \sum_{m=0}^M y_m^2 - 2W\sum_{m=0}^My_mz_m + W^2\sum_{m=0}^M z_m^2}{2\sigma^2} + K \\
				
				& = & -Slog(\sqrt{2\pi}) - Slog(\sigma) - \dfrac{\sum_{n=0}^N(y_n - Wz_n)^2 + \sum_{m=0}^M y_m^2 - 2W\sum_{m=0}^My_m\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_m] + W^2\sum_{m=0}^M \mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2]}{2\sigma^2} + K \\

\end{eqnarray}
$$


### Update Parameter - $\sigma$

$$
\dfrac{\partial Q}{\partial \sigma} = -\dfrac{S}{\sigma} + \dfrac{\sum_{n=0}^N(y_n - Wz_n)^2 + \sum_{m=0}^M y_m^2 - 2W\sum_{m=0}^My_m\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_m] + W^2\sum_{m=0}^M \mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2]}{\sigma^3}
$$

$$
\begin{eqnarray}

0  & = & -\dfrac{S}{\sigma} + \dfrac{\sum_{n=0}^N(y_n - Wz_n)^2 + \sum_{m=0}^M y_m^2 - 2W\sum_{m=0}^My_m\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_m] + W^2\sum_{m=0}^M \mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2]}{\sigma^3} \\

S\sigma^2 & = &  \sum_{n=0}^N(y_n - Wz_n)^2 + \sum_{m=0}^M y_m^2 - 2W\sum_{m=0}^My_m\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_m] + W^2\sum_{m=0}^M \mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2] \\


\sigma  & = & \sqrt{\dfrac{\sum_{n=0}^N(y_n - Wz_n)^2 + \sum_{m=0}^M y_m^2 - 2W\sum_{m=0}^My_m\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_m] + W^2\sum_{m=0}^M \mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2]}{S}}
\end{eqnarray}
$$





### Update Parameter - W

$$
\dfrac{\partial Q}{\partial W} = -\dfrac{1}{2\sigma^2}\left[-2\sum_{n=0}^Ny_nz_n + 2W\sum_{n=0}^Nz_n^2 - 2\sum_{m=0}^My_m\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_m] + 2W\sum_{m=0}^M \mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2]\right]
$$

$$
\begin{eqnarray}
0 & = & -\dfrac{1}{2\sigma^2}\left[-2\sum_{n=0}^Ny_nz_n + 2W\sum_{n=0}^Nz_n^2 - 2\sum_{m=0}^My_m\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_m] + 2W\sum_{m=0}^M \mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2]\right] \\

0 & = & -2\sum_{n=0}^Ny_nz_n + 2W\sum_{n=0}^Nz_n^2 - 2\sum_{m=0}^My_m\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_m] + 2W\sum_{m=0}^M \mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2] \\

W\sum_{n=0}^Nz_n^2 + W\sum_{m=0}^M \mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2] & = & \sum_{n=0}^Ny_nz_n + \sum_{m=0}^My_m\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_m] \\

W & = & \dfrac{\sum_{n=0}^Ny_nz_n + \sum_{m=0}^My_m\mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_m]}{\sum_{n=0}^Nz_n^2 + \sum_{m=0}^M \mathbb{E}_{P(Z_M|Y_M,Z_N,\theta_{old})}[Z_M^2]}
\end{eqnarray}
$$



