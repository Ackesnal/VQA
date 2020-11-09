# VQA

This repo is used to record the research process of Visual Question and Answering (VQA). 

I have tried Answer-based methods on the [Bootstrap](https://github.com/Cadene/bootstrap.pytorch) framework. The highest accurary is approx. 68%.

For the following research, I will convert to [OpenVQA](https://github.com/MILVLG/openvqa) framework.

---

## Ideation

__1. Use transformer as a basic method.__
  
    Image --> V_i, Q_i, K_i, Question --> V_q, Q_q, K_q. 
    
    Similarity: Sim(I) = softmax(V_i$\cdot$K_q), Sim(Q) = softmax(V_q)
    
    

## Current Issue

1. To what extend the novelty should be?
