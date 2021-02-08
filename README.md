# Random Walk in Latent Space (RWiLS)

The project implements network change detection.

RWiLS employs network embedding model to transform temporal networks into the latent space, performs random walk in the latent space, and estimates Dirichlet distributions with the results of random walk.

Two approaches are included in the projects to detect the variation of the network structure. 

1.   The first one fixes the amount of data under consideration, and two methods are used to measure the difference between two probability distributions, which are KL-divergence and MDL-change statistics respectively. The latter method is provided in the following paper: \
Kenji Yamanishi and Kohei Miyaguchi, "Detecting gradual changes from data stream using MDL-change statistics," 2016 IEEE International Conference on Big Data (Big Data), Washington, DC, 2016, pp. 156-163. \
Furhter, a method to aggregate the following methods is provided in 
Y. Yonamoto, K. Morino and K. Yamanishi, "Temporal Network Change Detection Using Network Centralities," 2016 IEEE International Conference on Data Science and Advanced Analytics (DSAA), Montreal, QC, 2016, pp. 51-60.

2.   The second one adjusts the considered data automatically. The method is provided in the following paper: \
R. Kaneko, K. Miyaguchi and K. Yamanishi, "Detecting changes in streaming data with information-theoretic windowing," 2017 IEEE International Conference on Big Data (Big Data), Boston, MA, 2017, pp. 646-655.

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
> Fixes the number of data under consideration. Four models are implemented, GAE and DeepWalk, along with KL-divergence and MDL-change statistics, respectively. Aggregation model is also provided.
```
python3 main_sliding.py
```
> Adjust the number of considered data dynamically.
```
python3 main_adaptive.py
```


# Contributions
*   GAE: [https://github.com/tkipf/gae](https://github.com/tkipf/gae)
*   Dirichlet Estimation: [https://github.com/ericsuh/dirichlet](https://github.com/ericsuh/dirichlet)
