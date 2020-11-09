# VQA

This repo is used to record the research process of Visual Question and Answering (VQA). 

I have tried Answer-based methods on the [Bootstrap](https://github.com/Cadene/bootstrap.pytorch) framework. The highest accurary is approx. 68%.

For the following research, I will convert to [OpenVQA](https://github.com/MILVLG/openvqa) framework.

---

## Ideation

#### 1. Use transformer as a basic method
  
Image --> V<sub>i</sub>, Q<sub>i</sub>, K<sub>i</sub>

Question --> V<sub>q</sub>, Q<sub>q</sub>, K<sub>q</sub>. 
    
Similarity: Sim(I) = softmax(Q<sub>i</sub> K<sub>q</sub><sup>T</sup>), Sim(Q) = softmax(Q<sub>i</sub>V<sub>q</sub><sup>T</sup>)

#### 2. Add relational info

The coordinates might work

Can be added to the image processes part? Or during the multi head? Or used to refine the similarity part?

---

## Current Issue

#### 1. To what extend the novelty should be?

Just found that [MMNAS net](https://arxiv.org/pdf/2004.12070.pdf) has used the relation info in the multimodal transformer.
    
For image i and image j, assume their bounding boxes are denoted as {x<sub>i</sub>, y<sub>i</sub>, w<sub>i</sub>, h<sub>i</sub>} and {x<sub>j</sub>, y<sub>j</sub>, w<sub>j</sub>, h<sub>j</sub>}.

Then the relationships between them are denoted as

R<sub>i,j</sub> = {|x<sub>i</sub> - x<sub>i</sub>|/w<sub>i</sub>,  |y<sub>i</sub> - y<sub>j</sub>|/h<sub>i</sub>,  w<sub>j</sub>/w<sub>i</sub>,  h<sub>j</sub>/h<sub>i</sub>}

R<sub>j,i</sub> = {|x<sub>i</sub> - x<sub>j</sub>|/w<sub>j</sub>,  |y<sub>i</sub> - y<sub>j</sub>|/h<sub>j</sub>,  w<sub>i</sub>/w<sub>j</sub>,  h<sub>i</sub>/h<sub>j</sub>}

And the similarity for images in the self-attention process is computed

Sim(x<sub>i</sub>, x<sub>j</sub>) = softmax(Q<sub>j</sub>K<sub>i</sub><sup>T</sup> / <sqrt>d</sqrt> + R<sub>i,j</sub>) """V<sub>i</sub>
