# AI Engineer — Microscopy Image Analysis (MIA-AI)

## Take-Home Technical Challenge

### Background

3D electron microscopy (EM) datasets provide rich structural information about brain wiring. Segmenting neurons and their ultrastructure, particularly organelles such as mitochondria, is a key challenge in building connectome datasets. This challenge explores working with state-of-the-art EM image data alongside modern self-supervised learning (SSL) models, specifically DINO.

---

### Instructions

- Commit your work to a Git repository and include a README with setup and reproduction instructions.
- Ensure all data acquisition is done programmatically — no manual downloads or screenshots.
- For all tasks below, it is acceptable to work with 2D slices rather than full 3D volumes.

---

### Task 1 — Data Acquisition

Download a subset of EM image data from at least **two** datasets available in the [OpenOrganelle repository](https://openorganelle.janelia.org/organelles/mito). All downloads must be performed programmatically (e.g., using the relevant API or access libraries).

---

### Task 2 — Feature Extraction with DINO

Use a pre-trained [DINO model](https://github.com/facebookresearch/dinov3) to produce embeddings for each of your two downloaded EM datasets.

Address the following:

1. **Patch size selection:** Which patch size is best suited to capture mitochondrial ultrastructure in the embeddings? Justify your choice.
2. **Dense embeddings:** Propose a method for obtaining dense, per-pixel (or per-voxel) embeddings rather than per-patch embeddings. Implement your proposed method and compute dense embeddings for both datasets.

---

### Task 3 — Embedding-Based Retrieval & Visualization

Select the embeddings corresponding to a single mitochondrion and use them as a query to evaluate how well the embeddings capture semantic information of other mitochondria.

1. **Within-dataset retrieval:** Visualize how the query mitochondrion's embeddings compare to the embeddings of other mitochondria *within* each dataset.
2. **Cross-dataset retrieval:** Visualize how the query mitochondrion's embeddings compare to the embeddings of mitochondria *across* the two datasets.
3. **Multiple queries:** Describe how you would adapt your visualizations and retrieval strategy if you used multiple query mitochondria instead of one. What changes would you expect in the results?

---

### Task 4 — Proposal: Improving Mitochondria Detection with Minimal Fine-Tuning

All preceding tasks use off-the-shelf DINO weights with no domain-specific training. Propose a plan for improving mitochondria detection or segmentation performance while minimizing the number of trainable parameters. A detailed technical outline is sufficient. No implementation is needed. 
