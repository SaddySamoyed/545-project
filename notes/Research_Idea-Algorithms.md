# topic

Data-centric methods to improve CLIP-based multimodal representation learning

[CLIP (Contrastive Language–Image Pretraining)Links to an external site.](https://arxiv.org/abs/2103.00020) is a multimodal representation learning model developed by OpenAI. It learns to understand images and text jointly by aligning them in a shared embedding space. Using a large dataset of image-text pairs, CLIP trains a vision model (e.g., ResNet or Vision Transformer) and a text model (e.g., Transformer) to predict which image matches a given text description and vice versa, via InfoNCE contrastive objective. This contrastive learning approach allows CLIP to generalize well to various tasks, like image classification or zero-shot learning, without task-specific fine-tuning, leveraging its ability to connect visual and textual concepts.

Studies have shown that CLIP performance can be enhanced by increasing the training objective difficulty. CLIP model is forced to learn to represent more delicate features in the images when the negative pairs are harder to distinguish from the positive ones. Several data-centric methods have been developed to do this, including putting more similar image-text pairs into one batch ([NegCLIPLinks to an external site.](https://arxiv.org/pdf/2210.01936), [SimCLIPLinks to an external site.](https://openreview.net/pdf?id=NmNmlAEBAl)) or introducing hard negatives via data augmentation ([NegCLIPLinks to an external site.](https://arxiv.org/pdf/2210.01936), [CE-CLIPLinks to an external site.](https://arxiv.org/pdf/2306.08832), [SLVCLinks to an external site.](https://arxiv.org/pdf/2211.11733)). These methods are typically benchmarked on natural image datasets (e.g. Imagenet or MSCOCO) as well as harder datasets designed specifically for evaluation of the model’s understanding of compositionality ([AROLinks to an external site.](https://huggingface.co/datasets/gowitheflow/ARO-Visual-Attribution), [VALSELinks to an external site.](https://paperswithcode.com/dataset/valse), [VL-CheckListLinks to an external site.](https://github.com/om-ai-lab/VL-CheckList), [SugarCrepeLinks to an external site.](https://github.com/RAIVNLab/sugar-crepe)). In addition, simple data augmentation that improves data diversity (even with no increase in difficulty) was also known to improve performance (e.g. [LLM Text RewriteLinks to an external site.](https://arxiv.org/abs/2305.20088)).

 

In this project, students are encouraged to explore or design new data-centric methods or augmentations to improve CLIP performance on small training datasets.



# Research Idea

现在我有一个 idea 来 propose 这个项目:

对于每一张 input 的图片，我们都用 yolov 提取其中的关键成分，然后把这些成分去掉，然后和周围的环境做一个平滑化。

对于 N 个关键成分的图片，我们采用关键成分的幂集: 2^N 个元素。对每个关键成分都选择 留/不留 这两个选项 (这不会太多, 因为，每张图最多有 3 个左右的关键成分, 所以数据集大小最多被扩大 8 倍, 我们是在小数据集上训练, 因而成本不会增加太大)，然后使用 llm 对图片对应的文字进行修改.

我这个想法运用了 manifold hypothesis: 我相当于对每个图片以关键成分取 discrete topology, 而后在这些图片的 product topology 上取完整的元素. 根据 manifold hypothesis, 所有的同等大小的图片构成了一个 topological manifold, 并且根据经验, 应该是包含这些图片 with repect to discrete topology 的 product manifold 的, 因而这样取点更符合 manifold hypothesis 的想法, 让数据变得更完整.

根据你的经验, 这个 idea 的可行性如何, significance 如何呢？



现在, 我们





We aim to improve CLIP's perception of the relationship between composition and text. 





# GPT prompt

------

### **1. 可行性分析**

#### **(1) 关键成分去除与平滑化**

- 使用 **YOLOv** 提取关键成分，然后移除并填充背景，这是计算机视觉中常见的操作，如 **inpainting (图像修复)**。现有方法如 **Stable Diffusion inpainting** 或 **GAN-based techniques (如 EdgeConnect, DeepFill)** 可以帮助做到这一点。
- 难点：
  - 在图像平滑化后，**背景可能仍然包含上下文信息**，使得模型能间接推测原始关键成分。
  - 处理复杂场景可能会破坏整体语义，导致数据增强后的样本偏离真实分布。

#### **(2) 关键成分的幂集扩展**

- 如果一张图片有 **N 个关键成分**，就可以形成 2N2^N 张图片。在实际场景下，大多数图片关键成分较少 (N≈3)，最多 **8 倍的数据扩展**，对计算成本的影响较小。
- 这种扩展能增加数据多样性，使模型学习 **成分的组合性** (compositionality)，类似于 **NegCLIP** 提出的 **hard negative mining** 方法。
- 潜在问题：
  - 扩展后的数据是否仍然符合自然图像分布？比如，在街道场景中移除所有行人后，场景仍然合理吗？
  - 可能需要对增强后的数据进行**过滤**，防止产生不合理的样本。

#### **(3) LLM 对文本描述的修改**

- 用 LLM（如 GPT-4V、BLIP、PaLM-2）修改图像描述，使其与不同的图片版本匹配，这是很有效的策略，类似 **LaCLIP** 的 **语言数据增强** 方法。
- 挑战：
  - 需要保证 LLM 生成的文本能够 **保持原始语义**，同时匹配变化后的图像。
  - 可能需要额外的 **人类反馈** 或 **自动质量评估** 机制。

------

### **2. Significance（研究意义）**

你的方法属于 **data-centric methods**，在 **CLIP-based multimodal learning** 领域可能带来的贡献：

#### **(1) 增强模型的组合性理解**

- CLIP 等视觉语言模型在 **组合性推理**（compositional reasoning）上仍然有很大提升空间（参考 **NegCLIP, CE-CLIP**）。
- 你的方法可以让模型在多个不同的上下文下学习关键成分的影响，从而更好地理解图像-文本关系。

#### **(2) 提高数据集的完整性**

- 通过对关键成分的幂集增强，可以让 CLIP **更全面地学习 manifold 上的不同数据分布**，避免仅依赖单一数据模式。
- 使得数据增强过程更符合 **流形假设 (manifold hypothesis)**，增强数据的覆盖度。

#### **(3) 适用于小数据集场景**

- 你提到的 **“小数据集” 训练** 场景是 CLIP 微调的主要瓶颈之一（参考 **SimCLIP**）。
- 你的方法 **不会增加标注成本**，但可以显著增加数据集的多样性，提高小样本学习的能力。

------

### **3. 可能的改进方向**

为了增强你的方法的可行性，以下是几个优化方案：

1. **优化 YOLOv 提取关键成分的策略**
   - 是否可以用 **attention-based methods（如 DINO, SAM）** 来增强关键成分识别？
   - 如何确定哪些成分是真正的 “关键成分”？是否可以用 **human-in-the-loop** 标注一部分数据来优化提取策略？
2. **增强 LLM 的文本生成质量**
   - 是否可以用 **多种 LLM (如 GPT-4V, BLIP-2, PaLM-2)** 交叉验证文本生成的质量？
   - 是否可以用 **对比学习 (contrastive learning)** 方法确保生成文本和图像匹配度更高？
3. **对增强后的数据进行自动评估**
   - 计算增强前后图像的 **特征分布 (feature distribution shift)**，确保它们仍然符合自然图像分布。
   - 使用 **CLIP embedding 相似性** 来检测增强样本的合理性。

------

### **4. 结论**

你的想法 **很有创新性**，并且有理论支持，但在实现时需要：

1. 解决 **关键成分去除后的背景合理性问题**；
2. 确保 **LLM 生成的文本能准确描述变化后的图像**；
3. 设计一个 **数据质量控制机制**，防止生成低质量样本。

如果你能解决这些问题，你的方法可以 **显著提升 CLIP 在小数据集上的泛化能力**，在 **multimodal representation learning** 领域具有很高的研究价值！





























# Algorithm

Prior research suggests that, 对于每一对 correct (image, caption) example, 如果改写它的 caption, 如调整语序和更改成分, 可以把这个 caption 改成一个很相似但是错误的 hard negative. 而我们的 idea 有相似与扩展之处:  我们假想, 对于每一对 correct (image, caption) example, 如果把 image 中的某个关键成分去掉, 并保留剩余部分, 那么这张新产生的 image 和原先的 image 有较高的相似度, 但是却不符合 caption 的描述; 而对于去除关键词后的 caption, 和新产生的 image 则是正确的配对.

对于去除关键成分后产生的新 image, 我们称其为 dual image; 对于去除关键词产生的新 caption, 我们则称之为 dual text. 

具体的做法是: 我们选取一个较小的 hyperparameter $N$ (e.g. $N=3$), 利用  in-context learning capability of large language models, 提取出这个文字中的 $N$ 个 key components (if possible), 并将它们归类于 a few pretrained classes. 紧接着, 我们通过 RCNN 模型 (e.g. YOLO v8) 进行 image segmentation, 在图片上提取 $N$ 个关键成分, 生成 mask, 再通过 diffusion model 的 image inpainting 能力把这些关键成分从图上去掉, 生成 $N$ 张 dual images. 额外地, 我们可以通过 LLM 对原 caption 进行改写, 分别把每个关键词从文字中去除, 相对应地得到 $N$ 个 dual texts.

我们的 idea 的合理性可以建立在  manifold hypothesis 上, 











Data-centric methods to improve CLIP-based multimodal representation learning

A way to mine hard negatives:



Hyperparameter: N

For each correct image-sentence pair:

1. use llm to automatically get $N$ **key noun components** of the sentence (N=3 e.g)
2. caption the N main components in the picture by **yolov**
3. for all subsets of the $\{N \text{ components}\}$, i.e. the power set & the discrete topology of it, we remove the following caption, and smoothing the picture by **stable diffusion inpainting**

The original image is true, while the new $2^N-1$ images are false, with the same setence.

Since we learn of a small-scale dataset, for small $N$ (e.g. N =3), the enlarged dataset should be not too big (e.g. 8 times the size)



Our method take advantage of the **manifold hypothesis**: All images of the same size is learned on a embedding topological manifold.

Since the picture with all $N$ components is a point, all local manifolds of the picutures is a submanifold of the embedding manifold. And **our modified $2^N$ images should be on the local submanifold around the original image**, thus **enriching the sampling of the local submanifold around  the picture**. We conjecture that this approach increases the combinatorial inference ability and robustness of the model on training with small datasets

This is similar to the LaClip (LLm rewrite), but our idea is to modify the image rather than the sentence. 
Furthermore, we can apply LLm rewrite to give the true statement of the new $2^N-1$ images. This can fix the problem that we have too many false examples and too few true examples in our new dataset.

For example: The original sentence is "Here is an apple, a banana and a pineapple".

We obtain the critical components "apple", "banana" and "pineapple", and remove "apple" from the image, making the example false.

And we can use llm to automatically rewrite the sentences, making this sentence "Here is a banana and a pineapple", which is true.







Prior research suggests that, 对于每一对 correct (image, caption) example, 如果改写它的 caption, 如调整语序和更改成分, 可以把这个 caption 改成一个很相似但是错误的 hard negative. 而我们的 idea 有相似与扩展之处:  我们假想, 对于每一对 correct (image, caption) example, 如果把 image 中的某个关键成分去掉, 并保留剩余部分, 那么这张新产生的 image 和原先的 image 有较高的相似度, 但是却不符合 caption 的描述; 而对于去除关键词后的 caption, 和新产生的 image 则是正确的配对.

对于去除关键成分后产生的新 image, 我们称其为 dual image; 对于去除关键词产生的新 caption, 我们则称之为 dual text. 

具体的做法是: 我们选取一个较小的 hyperparameter $N$ (e.g. $N=3$), 利用  in-context learning capability of large language models, 提取出这个文字中的 $N$ 个 key components (if possible), 并将它们归类于 a few pretrained classes. 紧接着, 我们通过 RCNN 模型 (e.g. YOLO v8) 进行 image segmentation, 在图片上提取 $N$ 个关键成分, 生成 mask, 再通过 diffusion model 的 image inpainting 能力把这些关键成分从图上去掉, 生成 $N$ 张 dual images. 额外地, 我们可以通过 LLM 对原 caption 进行改写, 分别把每个关键词从文字中去除, 相对应地得到 $N$ 个 dual texts.

我们的 idea 的合理性可以建立在  manifold hypothesis 上, 









