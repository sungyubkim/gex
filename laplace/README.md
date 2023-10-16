# Design principle of ```laplace```

Laplace approximation involves two-steps.

* Curvature estimation

    * Curvature defines the local property of loss landscape around $\theta^*$.

    * The principal space of $H$ finds the direction in which NNs are most sensitive.

    * Therefore, we need to find the principal eigenspace of $H^{-1}$ to find the low-loss parameters.

* Sample generation