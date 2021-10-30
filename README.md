# Random Walk in Latent Space (RWiLS)

The project implements network change detection.

RWiLS employs network embedding model to transform temporal networks into the latent space, performs random walk in the latent space, and estimates Dirichlet distributions with the results of random walk.

To estimate the probability distribtions, one needs to decide the sample data for estimation. A window is referred to the temporal range where data are under estimation.

Two approaches are included in the projects to detect the variation of the network structure. 

1.   The first one fixes the window size, using constant amount to estimate probability distribbutions. \
Two methods are used to measure the difference between two probability distributions,  KL-divergence and MDL-change statistics respectively. The latter method is provided in [1]. \
Further, a method to aggregate the following methods is implemented based on [2].

2.   The second one adjusts the window size dynamically. The method used is provided in [3].

Also, a comparison that runs the algorithm without network embedding models is included.

Two network embedding models are chosen in the project, which are GAE and DeepWalk, respectively. The resource code can be found in 
*   GAE
*   DeepWalk


# Requirements
*   networkx
*   scikit-learn
*   scipy
*   tensorflow (in GAE implementation)
*   gensim (in DeepWalk implementation)


# Run the Codes
> The window size is pre-determined and fixed.
```
python3 main_sliding.py
```
> The window size is adjusted dynamically.
```
python3 main_adaptive.py
```


# Contributions
*   GAE: [https://github.com/tkipf/gae](https://github.com/tkipf/gae)
*   Dirichlet Estimation: [https://github.com/ericsuh/dirichlet](https://github.com/ericsuh/dirichlet)


# Reference
[1] Kenji Yamanishi and Kohei Miyaguchi, "Detecting gradual changes from data stream using MDL-change statistics," 2016 IEEE International Conference on Big Data (Big Data), Washington, DC, 2016, pp. 156-163

[2] Y. Yonamoto, K. Morino and K. Yamanishi, "Temporal Network Change Detection Using Network Centralities," 2016 IEEE International Conference on Data Science and Advanced Analytics (DSAA), Montreal, QC, 2016, pp. 51-60

[3] R. Kaneko, K. Miyaguchi and K. Yamanishi, "Detecting changes in streaming data with information-theoretic windowing," 2017 IEEE International Conference on Big Data (Big Data), Boston, MA, 2017, pp. 646-655