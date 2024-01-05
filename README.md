
link prediction problem as a binary classification problem(Cora, Pubmed, Citeseer)
----------------------------------------------
Link prediction is a task in graph and network analysis where the goal is to predict missing or future connections between nodes in a network. Given a partially observed network, the goal of link prediction is to infer which links are most likely to be added or missing based on the observed connections and the structure of the network. In this report, we evaluate some networks on link prediction task for Cora dataset. We compare models in some criteta such as time inference, ,auc , size of model and needed time for buliding model. 

## 1.GraphSAGE model

This model is consisting of two **GraphSAGE layers(from DGL Library)**, each computes new node representations by averaging neighbor information. GraphSAGE was developed by Hamilton, Ying, and Leskovec (2017) 


Similar to NLP tasks, within embedding, we can facilitate several machine learning tasks on the graph, such as node classification, prediction, etc. Graph SAGE (SAmple and aggreGatE) provides an inductive approach to learning a low dimensional embedding of nodes in graphs. 

Similar to other GNN, GraphSAGE also operates iteratively in two stages,**aggregate and update,** and nothing much needs to be explained here. Its aggregate and update similar to other message passing layers and computed as other models. However, GraphSAGE is computationally efficient as it only requires a fixed-size neighborhood sampling instead of considering all nodes in the graph. 

It was formulates the link prediction problem as a binary classification
problem as follows:

-  Treat the edges in the graph as *positive examples*.
-  Sample a number of non-existent edges (i.e. node pairs with no edges
   between them) as *negative* examples.
-  Divide the positive examples and negative examples into a training
   set and a test set.
-  Evaluate the model with any binary classification metric such as Area
   Under Curve (AUC).


## 2.The NESS model
It is a framework for learning node embeddings from static subgraphs using a graph autoencoder (GAE) in a transductive setting. A GAE is a neural network that encodes the graph structure and features into a low-dimensional latent space, and then decodes the latent space to reconstruct the original graph. A transductive setting is a learning scenario where the model is trained and tested on the same set of nodes, but different sets of edges.

### Requirement

- **Static subgraph** is a subgraph if it does not change over time, and sparse if it has fewer edges than the original graph. A subgraph has non-overlapping edges with another subgraph if they do not share any common edges.


- **RES** is a method that randomly assigns each edge in the training graph to one of the subgraphs, such that each subgraph contains a fraction of the original edges. The nodes in each subgraph are the same as the original graph, but the edges are different. This way, the subgraphs capture different aspects of the graph structure and features, and can be used to learn node embeddings using a GAE.


<p align="center">
  <img width="30%" src="https://app.gemoo.com/share/image-annotation/601999289580212224?codeId=M0BAg8rnVlrEo&origin=imageurlgenerator" />
</p>

### Model



The NESS model consists of two steps: 
- Partitioning the training graph into subgraphs using random edge split (RES) during data pre-processing
- Aggregating the node representations learned from each subgraph to obtain a joint representation of the graph at test time.

**In the first step**, the NESS model splits the large graph into **train, validation, and test sets**, based on the edges. Then, it further divides the training set into multiple static, sparse subgraphs with non-overlapping edges using RES.



**In the second step**, the NESS model feeds each subgraph to the same encoder (i.e. parameter sharing) to get their corresponding latent representation. The encoder is usually based on a graph neural network (GNN) that aggregates the neighborhood information of each node. Then, the NESS model reconstructs the adjacency matrix of the same subgraph by using the inner product decoder. The decoder is a simple function that computes the dot product of the node embeddings to get the scores for the links. Thus, for the kth subgraph, we have: Encoder: $z_k = E(X,ak)$, and Decoder: $\widehat{a_k}=\sigma(z_k z_k^T)$, where $X$ is the node feature matrix, ak is the adjacency matrix, $z_k$ is the latent representation, $\widehat{a_k}$ is the reconstructed adjacency matrix, and $\sigma$ is the sigmoid function.


From above **Figure**, we see that training the model with a static subgraph emulates the same transductive learning setting, using the larger graph. Hence, NESS turns the standard transductive setting of GAE to multiple transductive settings, one for each subgraph. This allows the NESS model to learn more diverse and robust node embeddings, and improve the link prediction performance of GAEs.

## 3.WalkPool for link prediction by subgraph classification

The structure of the WalkPool model consists of four main steps:

**1. Feature extraction:** This step extracts node features from the original graph, either in an unsupervised or supervised way. For unsupervised features, the authors use DeepWalk , node2vec, and GraphSAGE For supervised features, the authors use GCN and GAT.

**2. Latent graph construction:** This step constructs a latent graph that captures the predictive features of the original graph. The latent graph is obtained by applying attention to the learned node features, which can be either unsupervised or supervised by a graph neural network. The attention mechanism assigns higher weights to the nodes that are more relevant to the target link.

**3. Walk probability pooling:** This step computes the walk probabilities of the paths connecting the target nodes in the latent graph. The walk probabilities are obtained by multiplying the transition probabilities of the latent graph, which are derived from the attention weights. The walk probabilities are then pooled to form a feature vector that summarizes the putative link.

**4.Link classification:** This step feeds the feature vector to a classifier to predict the link label. The classifier can be either a linear layer or a multi-layer perceptron.

<img src="wp-illustration.png" width="800" height="800">

## Evaluation
|Dataset|Model       |Epoch| Time-inference|Test Accuracy|
|--     |--          |--   |---            |         --  |
|Cora   |GraphSage   |100  |0.032 ± 0.003  |0.872 ± 0.007|
|Cora   |Ness        |200  |0.057 ± 0.007  |0.940 ± 0.000|
|Cora   |WalkPooling |5    |3.884 ± 0.176  |0.919 ± 0.004|



### Requirements

    python==3.10.6
    dgl==1.1.2
    pytorch==2.0.1+cpu
    torch-scatter==2.1.1+pt20cpu
    torch-sparse==0.6.17
    torch-geometric==2.4.0



