# VQA - CSIRO Data61 - Xuwei Xu

This repo is used to record the research process of Visual Question and Answering (VQA). It is built upon the [OpenVQA](https://github.com/MILVLG/openvqa) framework.

I have tried Answer-based methods on the [Bootstrap](https://github.com/Cadene/bootstrap.pytorch) framework. The highest accurary is approx. 68%.

For the following research, I will convert to [OpenVQA](https://github.com/MILVLG/openvqa) framework.

---

## Ideation

#### 1. Use transformer as a basic method (HAS BEEN WIDELY USED)
  
Image --> V<sub>i</sub>, Q<sub>i</sub>, K<sub>i</sub>

Question --> V<sub>q</sub>, Q<sub>q</sub>, K<sub>q</sub>. 
    
Similarity: Sim(I) = softmax(Q<sub>i</sub> K<sub>q</sub><sup>T</sup>), Sim(Q) = softmax(Q<sub>i</sub>V<sub>q</sub><sup>T</sup>)

<br>
<br>

#### 2. Add relational info (HAS BEEN USED, BUT CAN BE APPROVED)

The coordinates might work

Can be added to the image processes part? Or during the multi head? Or used to refine the similarity part?

<br>
<br>

#### 3. Use POS tags

To identify the part of speech (POS) of each word. 

3.1 Attempt to give part of image (POI) maybe? That is, assign different labels to different objects of an image.

3.2 Using POS and Bounding Boxes to extract vision attention


#### 5. Derive relationship via b-box

Calculate the type of relationship between two objects by their bounding boxes and feature maps.


<br>
<br>

#### 4. VQA-version YOLO?

Say, a one-step only end-to-end model for VQA?

Initial idea: 
  
    1. Question attention vector will be used to compute the kernel for each layer in CNN.
    2. Question attention vector will be attended by transformer (self-attention).
    3. For each layer, the original image is convoluted by the kernel generated from question, the question attention is attended by self-attention layer.
    
<br>
<br>

#### 6. Use postional similarity

For input image and question,

1) Key from question that has shape __d<sub>y</sub> * n__

2) Value from question that has shape __d<sub>y</sub> * n__

3) Query from image that has shape __d<sub>x</sub> * n__

4) Bounding box embedding from image that has shape __d<sub>x</sub> * m__

**Firstly**, calculate the (cosine or etc) similarity between each pair of bounding boxes to get Pos_Sim that has shape __d<sub>x</sub> * d<sub>x</sub>__.

Pos_Sim<sub>i,j</sub> represents the similarity between item i and j.

**IMPORTANT**, it should be directional, i.e.  Pos_Sim<sub>i,j</sub> != Pos_Sim<sub>j,i</sub>

**Secondly**, calculate Sim = Query * Key<sup>T</sup> that has shape __d<sub>x</sub> * d<sub>y</sub>__

**Thirdly**, calculate the refined similarity by **Pos_Sim * Sim** whose result has shape __d<sub>x</sub> * d<sub>y</sub>__

<br>
<br>

---

## Current Issue

#### 1. To what extend the novelty should be? (SOLVED)

Just found that [MMNAS net](https://arxiv.org/pdf/2004.12070.pdf) has used the relation info in the multimodal transformer.
    
For image i and image j, assume their bounding boxes are denoted as {x<sub>i</sub>,&nbsp; y<sub>i</sub>,&nbsp; w<sub>i</sub>,&nbsp; h<sub>i</sub>} and {x<sub>j</sub>,&nbsp; y<sub>j</sub>,&nbsp; w<sub>j</sub>,&nbsp; h<sub>j</sub>}.

Then the relationships between them are denoted as

R<sub>i,j</sub> = {|x<sub>i</sub> - x<sub>i</sub>|/w<sub>i</sub>,&nbsp;&nbsp;  |y<sub>i</sub> - y<sub>j</sub>|/h<sub>i</sub>,&nbsp;&nbsp;  w<sub>j</sub>/w<sub>i</sub>,&nbsp;&nbsp; h<sub>j</sub>/h<sub>i</sub>}

R<sub>j,i</sub> = {|x<sub>i</sub> - x<sub>j</sub>|/w<sub>j</sub>,&nbsp;&nbsp;  |y<sub>i</sub> - y<sub>j</sub>|/h<sub>j</sub>,&nbsp;&nbsp;  w<sub>i</sub>/w<sub>j</sub>,&nbsp;&nbsp; h<sub>i</sub>/h<sub>j</sub>}

And the similarity for images in the self-attention process is computed as:

Sim(x<sub>i</sub>, x<sub>j</sub>) = softmax(Q<sub>j</sub>K<sub>i</sub><sup>T</sup> / <span>&#8730;</span>d + R<sub>i,j</sub>)

Sim(x<sub>j</sub>, x<sub>i</sub>) = softmax(Q<sub>i</sub>K<sub>j</sub><sup>T</sup> / <span>&#8730;</span>d + R<sub>j,i</sub>)

**In this case, is it still novel to use the spatial relation information?**

```
Answer:

As long as in different aspect. 
```

<br>
<br>


#### 2. Dimension does not match (SOLVED)

In [MMNAS net](https://arxiv.org/pdf/2004.12070.pdf), it introduced guided attention (GA)

Query **Q** is calculated from question input, and **Q** has dimension **m * d**

Key **K** is calculated from image input, and **K** has dimension **n * d**

Value **V** is calculated from image input, and **V** has dimension **n * d**

Relation **R** is calculated from bounding box, and **R** has dimension **n * n**

Similarity **S** is calculated from **Q**, **K** and **R** via **S<sub>GA</sub>** = softmax(**QK**<sup>T</sup>), **S<sub>RSA</sub>** = softmax(**QK**<sup>T</sup> + **R**)

For **S<sub>GA</sub>**, it has dimension **m * n**

Attended value **V'** is calculated via **V'** = **S<sub>GA</sub>V**, and it has dimension **m * d**
 
However, the initial dimension of **V** is **n * d**. Thus, the dimension is consistent through the process.
 
**Why? and how?**

```
Reason:

The output has the same size as Query.

It uses the image input as Query, and question input as Key and Value. Then the output has the same dimension as the image.

Q(uery) is derived from image features and has dimension n*d

K(ey) is derived from question attentions and has dimension m*d

V(alue) is derived from question attentions and has dimension m*d
```

<br>
<br>

#### 3. Why is it available to use image input as Query only? (SOLVED)

softmax(**QK**<sup>T</sup> + **R**) only returns the similarity figures.

The original image information will be lost by computing softmax(**QK**<sup>T</sup> + **R**)**V** because the **V** value matrix is computed from the question input.

```
Reason:

This step is to reconstruct the image feature by the question attention.

For each feature vector (i.e. an object), this unit will generate a query vector Q.

For each question attention, this unit will generate a key vector K and a value vector V.

Then it compares the Q and K to obtain similarities between the current object and every question vector.

The similarity result will be used to aggregate the question vector.

The aggregation result will be the reconstructed object feature vector.

For example, if the question feature has 3 vectors, each vector has 512 elements, corresponding to (cat, dog, bird)

If the object is dog.

Then the similarity might be (0.1, 0.85, 0.05).

In this case, the image feature vector is aggregated / reconstructured from the question vector.

```
<br>
<br>

#### 4. About applying for MPhil 

Whether Lars would like to be my supervisor? So that I can submit the application.

If Lars would like to, can we settle a time for discussion (about ideas and plans)?

What should I write in the proposal and where should I start?

```
Answer:

1. Talk with Moshiur to see if he would like to be the academic supervisor

2. Discuss about the research topic

3. Confirm with HDR Dean to see the graduate standards / criterion / requirements

4. Write research proposal and find three referee.

5. Apply
```

<br>
<br>

#### 5. To see whether different embeddings affect the performance.

In the original MCAN/MMNASNET, the question feature is directly derived from a LSTM layer with word embeddings from GloVe, the image feature is extracted by Faster-RCNN

No positional embedding is added to the question feature, unlike BERT using positional embedding or Transformer using positional encoding.

The bounding boxes are embedded to the same size of the image and concatenate to the end of image feature.

The current issues are:

    1. Whether concatenation is better than aggregation
    
    2. Whether positonal embedding (or even other embeddings) is better
    
    3. Whether training from scretch is better

---

## Findings

#### 1. Transformer-based model is faster than bilinear-fusion-based model.
