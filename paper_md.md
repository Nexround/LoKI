# LoKI: Low-damage Knowledge Implanting of Large Language Models

# Runyu Wang 1, Peng Ping 2* , Zhengyu Guo 3, Xiaoye Zhang 4, Quan Shi 2, Liting Zhou 5, Tianbo Ji 2

> 1

School of Information Science and Technology, Nantong University

> 2

School of Transportation and Civil Engineering, Nantong University

> 3

South China University of Technology

> 4

China Southern Power Grid Company Limited

> 5

Dublin City University 2430310032@stmail.ntu.edu.cn, {pingpeng, sq, jitianbo }@ntu.edu.cn, 202264700027@mail.scut.edu.cn, xiaoyz@whu.edu.cn, liting.zhou@dcu.ie

Abstract

Fine-tuning adapts pretrained models for specific tasks but poses the risk of catastrophic forgetting (CF), where criti-cal knowledge from pretraining is overwritten. To address the issue of CF in a general-purpose framework, we propose

Lo w-damage Knowledge Implanting ( LoKI ), a parameter-efficient fine-tuning (PEFT) technique that utilizes recent mechanistic understanding of how knowledge is stored in transformer architectures. We compare LoKI against state-of-the-art PEFT methods in two real-world fine-tuning sce-narios. The results show that LoKI demonstrates signif-icantly better preservation of general capabilities. At the same time, its task-specific performance is comparable to or even surpasses that of full parameter fine-tuning and these PEFT methods across various model architectures. Our work bridges the mechanistic insights of LLMs’ knowledge stor-age with practical fine-tuning objectives, enabling an effec-tive balance between task-specific adaptation and the reten-tion of general-purpose capabilities.

Code — https://github.com/Nexround/LoKI

Extended version — https://arxiv.org/abs/2505.22120

# Introduction

Transformer-based language models (Vaswani, Shazeer et al. 2017; Petroni, Rockt¨ aschel et al. 2019) accumulate extensive world knowledge during pretraining (Radford, Narasimhan et al. 2018; Brown, Mann et al. 2020), which becomes implicitly embedded in their parameters. Owing to their broad generalization capabilities, they serve as a strong foundation for downstream task adaptation, prompting sig-nificant efforts to develop increasingly efficient fine-tuning methods (Prottasha, Chowdhury et al. 2025; Xin, Yang et al. 2025). However, the process of fine-tuning pretrained mod-els for downstream tasks is often accompanied by catas-trophic forgetting (CF) (Kotha, Springer et al. 2024; Luo, Yang et al. 2025), a phenomenon where the model loses previously acquired capabilities after fine-tuning. Recent studies have pointed out that conventional fine-tuning ap-proaches (Prottasha, Chowdhury et al. 2025; Xin, Yang et al.

> *

Corresponding author. Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved. fine -tune on

> downstream task

Analysing Selecting Implanting  

> KVA Layer -Balanced
> Strategy
> Layer -Balanced
> Strategy
> higher contribution
> node
> higher contribution
> node
> selected
> node
> selected
> node
> down -proj
> matrix
> down -proj
> matrix active active frozen frozen

Figure 1: Schematic illustration of the staged fine-tuning process in LoKI. 2025) typically perform indiscriminate updates across mod-ules within transformer architectures, oblivious to these cru-cial knowledge-storing weights in general tasks. Such un-constrained optimization may perturb crucial memory traces (Cohen, Geva et al. 2023; Petroni, Rockt¨ aschel et al. 2019), leading to irreversible knowledge loss and degraded general-ization performance (Li, Ding et al. 2024; Huang, Cui et al. 2024). Extensive research has investigated the internal struc-ture of large language models (LLMs) (Li, Li et al. 2024; Meng, Sharma et al. 2023a), revealing that model-internal knowledge is both identifiable and editable via targeted in-terventions such as knowledge localization and rewriting (Geva, Schuster et al. 2021; Meng, Sharma et al. 2023a). Moreover, the inherent redundancy in LLMs has been shown to support techniques like sparsification and parameter prun-ing (He, Zhou et al. 2022; Frankle and Carbin 2019), en-abling more efficient representation without significantly de-grading performance (Lasby, Golubeva et al. 2024). How-ever, these findings have yet to be effectively integrated into the field of model fine-tuning. These findings lead us to hypothesize that it is possible to identify low-contributing weights in pretrained models, which can then serve as ca-pacity for injecting new, task-specific knowledge with mini-mal impact on existing competencies. Based on this assump-tion, we propose Lo w-damage Knowledge Implanting, as LoKI , a parameter-efficient fine-tuning method that lever-ages insights from interpretability studies on the internal knowledge storage mechanisms of LLMs. Aiming to miti-gate catastrophic forgetting, LoKI consists of three stages:

analyzing , selecting , and implanting (see Figure 1). In the

analyzing stage, we introduce Knowledge Vector Attribu-tion (KVA), a gradient-based attribution technique (Sun-dararajan, Taly et al. 2017) that evaluates the contribution of each vector in the down-projection matrix Wdown (see Background section for details) to the model’s pretrained behavior. In the selecting stage, motivated by the known interdependence between transformer layers (Sun, Pickett et al. 2025a)—especially in the progressive refinement of knowledge (Geva, Bastings et al. 2023; Dai, Dong et al. 2022a)—we propose the Layer-Balanced Strategy, which ensures that new knowledge aligns with the model’s hier-archical structure. Leveraging KVA results, this strategy en-forces an equal quota of trainable parameters per layer by decomposing each Wdown into two subsets: WS (active) and

W\S (frozen). This yields a layer-balanced trainable subset

WS . In the implanting stage, we freeze all model parame-ters except WS , which is updated via fine-tuning. Compared to existing fine-tuning approaches, LoKI offers several ma-jor advantages. • Superior balance between CF and task-specific per-formance. Fine-tuning LLMs of two representative sizes with LoKI effectively mitigates CF, outperforming state-of-the-art PEFT methods while maintaining strong task performance. • Intrinsically parameter-efficient. LoKI updates only a subset of the model’s original parameters and allows for explicit control over the number of trainable weights. • Synergistic with other tuning methods. In addition to adapting directly for downstream tasks, we demonstrate that LoKI can be combined with existing parameter-efficient tuning techniques such as Low-Rank Adaptation (LoRA) (Hu, Shen et al. 2022), further reducing the num-ber of trainable parameters. By allocating updates to carefully selected weights, LoKI provides a competitive approach to sustainable LLM cus-tomization, enabling models to evolve while preserving their core capabilities. In summary, our main contributions are: • We introduce LoKI, a fine-tuning framework for LLMs that primarily aims to mitigate CF during adaptation to downstream tasks. • We propose KVA, a technique to quantify the contri-bution of individual knowledge vectors (defined in the Background section) to the model’s stored representa-tions. Furthermore, using this method, we uncover a sur-prising phenomenon: in transformer models, knowledge vectors with both globally high and low contributions tend to be densely located in similar layers. • Building on an understanding of the hierarchical orga-nization of knowledge in transformers, we develop the Layer-Balanced Strategy for allocating trainable weights. Our experiments show that this strategy is critical to LoKI’s ability to preserve pre-trained capabilities while learning new tasks.

# Related Works

Catastrophic Forgetting CF has long been recognized as a critical challenge in neural networks (French 1993; Kemker, McClure et al. 2018a), and recent studies have be-gun to analyze its manifestation specifically in LLMs (Li, Ding et al. 2024; Kotha, Springer et al. 2024). Represen-tative methods include: Replay-based methods (de Mas-son d’Autume, Ruder et al. 2019; Lopez-Paz and Ranzato 2017; Chaudhry, Ranzato et al. 2019) interleave data from previous tasks during training to reinforce prior knowledge. Regularization-based methods (Li and Hoiem 2018; Kirk-patrick, Pascanu et al. 2016; DENG, Chen et al. 2021) con-strain parameter updates to prevent drift from previously important weights. However, most of the previous methods generally treat knowledge retention as a black box, rarely considering how pretrained LLMs organize and store knowl-edge internally.

Parameter-Efficient Fine-Tuning Recent advances in PEFT aim to mitigate CF by preserving pretrained knowl-edge while enabling task adaptation. For example, Zhu, Sun et al. refines a small subset of critical parameters to retain performance on original tasks. However, it requires task-specific parameter selection. Some methods structurally de-couple knowledge to resist CF. CorDA (Yang, Li et al. 2024) freezes dominant singular directions, assumed to en-code general knowledge, while adapting residual subspaces. LoRASculpt (Liang, Huang et al. 2025) uses magnitude-based masking and conflict-aware regularization to con-strain LoRA (Hu, Shen et al. 2022) updates within critical knowledge regions. Orthogonal subspace approaches like O-LoRA (Wang, Chen et al. 2023)and LoRI (Zhang, You et al. 2025) reduce task interference by isolating updates in mutually orthogonal directions, effective in multi-task sce-narios but less focused on preserving original capabilities. In contrast, LoKI proposes a one-time knowledge localiza-tion strategy, directly quantifying parameter contributions to general knowledge, enabling scalable and effective CF resis-tance across diverse downstream tasks.

Knowledge Locating and Editing ROME (Meng, Bau et al. 2022) revises individual factual associations by di-rectly modifying the FFN weight vectors. KN (Dai, Dong et al. 2022a) regulates the expression of specific facts by controlling the activation levels of identified neurons. Al-phaEdit (Fang, Jiang et al. 2025) further advances this line by applying zero-space projection to the output weight ma-trix of FFN layers to perform targeted edits. However, de-spite their effectiveness in editing specific knowledge ex-pressions, these methods have yet to be seamlessly inte-grated into parameter-efficient fine-tuning pipelines that aim to retain the model’s broad pretrained capabilities. More-over, compared to existing knowledge localization methods, KVA and the Layer-Balanced Strategy constitute a complete model analysis pipeline in practice. This combination tran-scends the limitation of focusing solely on specific factual expressions, evolving the underlying ideas into a general-purpose approach for model analysis. Background

We will begin this section by briefly reviewing transformer architecture and relevant research on knowledge storage in transformer-based language models, which will establish a foundation for our proposed method. Transformer-based language models are built from stacked layers, each consist-ing of two primary components: a multi-head self-attention (MHSA) mechanism (Vaswani, Shazeer et al. 2017) and a position-wise feed-forward network (FFN). While some studies have explored the role of MHSA in transformers (Voita, Talbot et al. 2019; Clark, Khandelwal et al. 2019; Vig and Belinkov 2019), a growing number of studies fo-cus on the FFN layers for knowledge localization and edit-ing (Geva, Schuster et al. 2021; Geva, Caciularu et al. 2022; Katz, Belinkov et al. 2024). For an input vector x ∈ Rdmodel ,a typical FFN without bias applies two linear transforma-tions with a non-linearity in between:

FFN( x) = Wdown σ Wup x (1) where Wup ∈ Rdff ×dmodel is the up-projection matrix,

Wdown ∈ Rdmodel ×dff is the down-projection matrix, and

σ(·) denotes an element-wise activation function. Recent studies suggest that FFN layers function as linear associa-tive memories, where the Wup acts as a collection of keys detecting input patterns, and the Wdown contains values cor-responding to interpretable concepts (Geva, Schuster et al. 2021; Geva, Caciularu et al. 2022). This view has been reinforced by both activation- and gradient-based analyses (Katz, Belinkov et al. 2024). Specifically, each output coor-dinate yj in the FFN can be viewed as a knowledge output node, computed as yj = v⊤

> j

σ(Wup , x ), where vj is the j-th row of Wdown and serves as a knowledge vector. On the other hand, while single-layer FFN reveals this memory be-havior, deeper inspection shows a hierarchy: lower FFN lay-ers capture surface-level patterns, whereas upper layers en-code higher-level semantics (Geva, Bastings et al. 2023; Dai, Dong et al. 2022a; Tan, Zhang et al. 2024; Katz, Belinkov et al. 2024). The final output distribution of transformer-based language models is gradually constructed in a bottom-up fashion (Sun, Pickett et al. 2025a; Tenney, Das et al. 2019; Wallat, Singh et al. 2020). Such insights have in-spired a wave of model editing techniques that target the FFN layers—particularly Wdown —to insert, update, or erase factual knowledge without extensive retraining (Geva, Caci-ularu et al. 2022; Dai, Dong et al. 2022b; Meng, Sharma et al. 2023b). Motivated by the success of these approaches, our study focuses on implanting task-specific knowledge through FFNs, which we consider the main site for im-planting new knowledge into the model. Importantly, prior research reveals that LLMs exhibit considerable parame-ter redundancy, especially within FFNs (Kobayashi, Kurib-ayashi et al. 2024; Sanh, Wolf et al. 2020a; Zhang, Lin et al. 2022; Sanh, Wolf et al. 2020b; Frantar, Alistarh et al. 2023). Numerous studies show that substantial portions of model weights can be pruned or restructured with minimal per-formance loss on general tasks (Frankle and Carbin 2019; Michel, Levy et al. 2019; Sanh, Wolf et al. 2020b; Fran-tar, Alistarh et al. 2023). Based on the previous research mentioned above, we hypothesize that these low-impact pa-rameters in FFN layers can be repurposed to encode new task-specific knowledge without excessively degrading the model’s original capabilities. Our experimental results pro-vide evidence supporting this hypothesis.

# Low-damage Knowledge Implanting

We introduce LoKI, a three-stage framework— analyzing ,

selecting , and implanting —designed to edit transformer models with minimal disruption, thereby mitigating CF (see Figure 1). Below, we briefly outline each stage before pro-viding a detailed explanation: 1. Analyzing : Evaluate the contribution of each knowledge vector to general tasks. 2. Selecting : Choose trainable knowledge vectors within each FFN based on the analysis results. 3. Implanting : Train the chosen vectors to incorporate task-specific knowledge.

## Analysing

We start this subsection by introducing Knowledge Vector Attribution (KVA) , an attribution method inspired by (Hao, Dong et al. 2021; Dai, Dong et al. 2022a), designed to eval-uate the contribution of individual knowledge vectors to the model’s performance. Next, we illustrate the workflow of the analysis stage—in other words, how KVA is utilized within the LoKI framework.

Knowledge Vector Attribution KVA is a computational approach based on Integrated Gradients (IG) (Sundararajan, Taly et al. 2017) that measures the contribution of knowl-edge vectors to specific output logits.

IG i(x) = ( xi − x′

> i

)

Z 10

∂F (x′ + α(x − x′))

∂x i

dα, (2) where F is the network function, x′ is a baseline (we use

x′ = 0), and α ∈ [0 , 1] is an interpolation coefficient that defines the integration path from baseline to input. IG at-tributes predictions to input features by integrating gradients along the straight-line path from baseline x′ to input x. To trace knowledge flow through all layers of a transformer on input sequence x, let L denote the target logit, hl−1 repre-sent the hidden state from the previous layer. At layer l, the FFN first computes pre-activations

zl = W (l)up hl−1, ul = σ(zl), zl,j = ul,j W (l)down ,j ·.

(3) As mentioned above, we treat FFNs as collections of key-value memory pairs. To attribute the contribution of each knowledge vector to the final output of an LLM, we define the layer-wise, path-integrated attribution of node j by

Attr l,j (x) =

Z 10

∂L α zl,j



∂zl,j

dα, (4) which aggregates the total gradient flowing through knowl-edge vector j as its contribution is scaled from zero up to its actual activation. We approximate the above integral via Riemann approximation with m equally-spaced steps (we set the step m = 7 for any model, and further discussion on the setting of m can be found in Appendix A):

Attr l,j (x) ≈ 1

m

> m

X

> k=1

∂L  km zl,j



∂zl,j

. (5) During each sample, KVA computes and stores Attr l,j (x)

for every layer l and every knowledge output node j. This yields a complete attribution log that can be analyzed post hoc to assess the contribution of each knowledge vector dur-ing inference.

Scoring Knowledge Vectors Pretrained LLMs exhibit a wide range of capabilities, including world knowl-edge, common-sense reasoning, and instruction follow-ing. While it is difficult to precisely determine the full scope of an LLM’s internal knowledge, it can be ap-proximated through extensive textual input (Meng, Sharma et al. 2023a). To evaluate the contribution of individual knowledge vectors to general-purpose performance—and to identify those with minimal impact on core capabili-ties—a critical factor is the choice of dataset for quan-titative analysis. To ensure broad domain coverage, we apply KVA to the Massive Multitask Language Understanding (MMLU) benchmark (Kemker, Mc-Clure et al. 2018b). MMLU is ideal because: • it has 57 diverse subjects (STEM, humanities, profes-sional, etc.), which covers a broad spectrum of general-knowledge tasks, ensuring the evaluation score of vectors is not domain-specialized; • its standardized prompts and evaluation make attribu-tions comparable across tasks; • high performance on MMLU correlates with real-world language understanding, so protecting original perfor-mance is measurable. This process, executed on an RTX4090 GPU using Llama3.1-8B-Instruct (Dubey et al. 2024), takes an aver-age of 9.69 seconds per sample. In parallel, we apply KVA to the full MMLU dataset. Results show that the final se-lected nodes—based on the Layer-Balanced Strategy (de-tailed later)—overlap by 97.57% between the reduced set and the full dataset. This confirms that our sampling strategy substantially reduces computational cost while preserving high fidelity in attribution outcomes. Notably, KVA needs to be performed only once per model, independent of down-stream tasks. That is, this computational overhead is in-curred a single time per model. Implementation details and additional discussion are provided in Appendix A.

## Selecting

As shown in Figure 2, the evaluation results of KVA reveal significant heterogeneity in the distribution of knowledge output nodes across different layers, regardless of their con-tribution level. This uneven distribution phenomenon among nodes with high contributions aligns with the findings of Meng et al. (2022; 2023a) and Dai et al. (2022b). No-tably, our results uncover a novel phenomenon: both high-1

> 3
> 5
> 7
> 9
> 11
> 13
> 15
> 17
> 19
> 21
> 23
> 25
> 27
> 29
> 31 1
> 3
> 5
> 7
> 9
> 11
> 13
> 15
> 17
> 19
> 21
> 23
> 25
> 27
> 29
> 31

Figure 2: Heatmaps of the top 5% KVA results across all 32 layers of Llama3.1-8B-Instruct. The vertical axis de-notes node indices, and the horizontal axis denotes layer in-dices. The upper (red-tinted) heatmap illustrates the distri-bution of high-contribution node positions, while the lower (blue-tinted) heatmap illustrates the distribution of low-contribution node positions. Color intensity (log-scale) re-flects the density of nodes within each category, with darker colors indicating higher density. Heatmaps for additional models are provided in Appendix B. and low-contribution nodes are densely concentrated in the same layers. This suggests a potential structural relation-ship in the placement of knowledge vectors across differ-ent attribution levels. We believe this observation may of-fer new insights into inter-layer knowledge organization in transformer architectures. As our current focus is on lever-aging the analysis results for guiding fine-tuning, we leave a deeper investigation of this phenomenon to future work. These findings shed a critical light on fine-tuning strategies: the layer-wise distribution of knowledge is not random but structurally biased. Consequently, na¨ ıve parameter-efficient approaches that update only a limited number of layers may inadvertently disrupt the intrinsic knowledge hierar-chy of transformers (Sun, Pickett et al. 2025b; Geva, Caci-ularu et al. 2022). Our experiments show that such imbal-anced parameter allocation exacerbates catastrophic forget-ting (see the Ablation Studies section). Furthermore, Hase et al. (2023) have found a significant challenge: even when factual knowledge resides in specific layers, such as the mid-layer FFNs, modifying weights in distant layers, especially the earlier ones, can effectively ”override” the information flow downstream.

Layer-Balanced Strategy Based on the previous consid-erations, we propose the Layer-Balanced Strategy, which is a strategy that determines the trainable parameter posi-tions under the constraint of allocating the same number of parameters to each layer. This strategy aims to (a) ensure that newly implanted knowledge conforms to the hierarchi-cal relationship of the transformer model’s knowledge struc-ture, and (b) avoid disproportionately disturbing its inherent knowledge hierarchy. Let the model have L layers, where only parameters within each layer’s Wdown matrix are par-tially trainable, while all other parameters across all layers remain frozen. Each Wdown in layer l contains Dl knowl-edge output nodes (with Dl = D assumed identical across layers for simplicity). The hyperparameter q ∈ (0 , 100)

governs the percentage of trainable nodes selected from all

Wdown matrices. Given N inference samples, we denote the KVA result for node i in layer l on sample t as:

Attr (t)

> l,i

, i = 1 , . . . , D, t = 1 , . . . , N

The implementation proceeds as follows: 1. Quota Allocation: Calculate the total trainable slots:

T = q

100 ·

> L

X

> l=1

Dl = q

100 · LD (6) Allocate equal quotas per layer: kl =  TL

 , l =1, . . . , L.

2. Per-Sample Local Selection: For each sample t and layer l, select kl nodes with the smallest values:

S(t)

> l

=argsort ↓

> i∈{ 1,...,D }

 ˆAttr (t)

> l,i



[1 : kl] (7) 3. Frequency Aggregation: Tally selection frequencies across samples:

cl,i =

> N

X

> t=1

I



i ∈ S(t)

> l



(8) 4. Final Selection: For each layer l, select nodes with the highest frequencies:

Sl =argsort ↑

> i∈{ 1,...,D }

(cl,i )[1 : kl], S =

> L

[

> l=1

Sl (9) Resulting in a balanced set S with PLl=1 |S l| ≤ T .To empirically assess the effectiveness of the Layer-Balanced Strategy, we perform an ablation study comparing it with imbalanced baselines that na¨ ıvely select knowledge vectors with the highest or lowest global attribution scores as trainable parameters. Experimental details and results are provided in the Ablation Studies section.

## Implanting

Building upon the selected knowledge output nodes S =

SLl=1 Sl, we implement parameter-efficient fine-tuning through strategic decomposition of FFN layers. For each layer l’s down-projection matrix W (l)down , we partition the pa-rameters into two complementary subspaces:

W (l)down =

 WSl

W\S l



, (10) where WSl ∈ R|S l|× dff contains the weights correspond-ing to our selected low-contribution knowledge output nodes (from layer l’s quota Sl), and W\S l represents the remaining parameters. During training, we: • Keep W\S l frozen to preserve existing knowledge repre-sentations • Update only WSl actively to implant new knowledge This decomposition retains the mathematical formulation of the original layer while restricting parameter updates to the chosen subspaces. In addition, it transforms LoKI training into a module-wise process, allowing easy integration with existing training pipelines. For clarity, we refer to this im-plementation as LoKI Linear , and the readers can find its PyTorch code in Appendix C.

Incorporating LoRA It is worth noting that this imple-mentation allows for the incorporation of low-rank decom-position techniques (Hu, Shen et al. 2022; Liu, Wang et al. 2024; Zhang, Chen et al. 2023) into LoKI. Specifically, the trainable weights are parameterized as:

WSl = W (0)  

> Sl

+ ∆ WSl , ∆WSl = AlBl, (11) where W (0)  

> Sl

denotes the frozen base weights, and Al ∈

R|S l|× r , Bl ∈ Rr×dff are learnable low-rank matrices with rank r ≪ min( |S l|, d ff ). To verify the feasibility of this in-tegration, we experimented with the Experiment section.

# Experiments

To evaluate our proposed method, we focus on adapting models to the following two datasets: 1. ToolACE Function-Calling Dataset (Liu, Huang et al. 2024): This dataset contains 26,507 distinct APIs across various domains, designed to enhance the function-calling capabilities of LLMs. Consistent with the offi-cial model released by the dataset provider, we fine-tuned Llama3.1-8B-Instruct on this dataset. The details of the experiment setup can be found in Appendix D. 2. LB Reranker Dataset (Lightblue 2024): This multilin-gual dataset consists of 2.28 million query-text pairs an-notated with fine-grained relevance scores on a 1-7 scale, designed for training NLP-based retrieval models. It has an official full parameter fine-tuning model based on Qwen2.5-0.5B-Instruct (Qwen, Yang et al. 2025); we uti-lized the same base model on this dataset. The experi-ment setup for this task can be found in Appendix E. These publicly available datasets were specifically selected due to their demonstrated practical utility in real-world ap-plications and their ability to simulate realistic fine-tuning demand scenarios. Each experiment assesses the capabil-ity of LoKI to resist the CF phenomenon while acquiring task-specific performance. Notably, while our experiments involved only two model types, the proposed method is read-ily applicable to other model architectures.

Measuring Catastrophic Forgetting To systematically evaluate CF during fine-tuning, we assess the model’s retained general capabilities across six diverse bench-marks: TriviaQA (Joshi, Choi et al. 2017) (world knowl-edge), GSM8K (Cobbe et al. 2021) (mathematical reason-ing), HellaSwag (Zellers, Holtzman et al. 2019) and

WinoGrande (Sakaguchi, Bras et al. 2020) (commonsense understanding), HumanEval (Chen et al. 2021) (code gen-eration), and IFEval (Zhou, Lu et al. 2023) (instruction following). We define the average forgetting score across all bench-marks as: Avg = 100

> N

PNi=1  

> So
> i−St
> i
> So
> i

, where So

> i

and St

> i

denote the performance of the original and fine-tuned models on the

i-th benchmark, respectively, and N is the total number of benchmarks. A higher value of Avg indicates a greater de-gree of forgetting. Further details regarding the evaluation of these benchmarks can be found in Appendix F.

Task 1: ToolACE Function-Calling Dataset Based on the training results on this dataset, we first compare LoKI with existing SOTA methods in terms of downstream task adaptation. We evaluate fine-tuned models’ performance on the Berkeley Function Calling Leaderboard V3, comparing LoKI against LoRA (model provided by the dataset authors), DoRA (Liu, Wang et al. 2024), and PiSSA (Meng, Wang et al. 2024). As shown in Table 1, LoKI achieves the high-est overall accuracy when q=30 . Under the q=20 setting, LoKI exhibits the strongest multi-turn reasoning capabil-ity, with a success rate of 17.75% . Notably, all LoKI vari-ants consistently reduce the Irrelevance metric, suggesting that LoKI may help mitigate hallucination effects introduced during fine-tuning. Table 2 further reveals LoKI’s unique capability in preserving foundational model competencies. Compared to DoRA, LoKI ( q=30 ) not only outperforms in fine-tuned performance, but also reduces the average perfor-mance degradation percentage by 75% across six evalua-tion benchmarks. Relative to the original pretrained model, the degradation is as small as 1.23% . Notably, as the hyper-parameter q increases from 10 to 30—despite a substantial growth in the number of trainable parameters—the rate of performance degradation slows down. Additionally, we ex-perimented to investigate the potential of combining LoRA with LoKI (LoKI*( q=30 )). As anticipated, the number of trainable parameters in the model significantly decreased when integrated with LoRA, showing a reduction of 97.16% compared to LoKI( q=30 ). Importantly, this combination did not visibly compromise LoKI’s ability to resist catastrophic forgetting. we believe that this combination holds significant promise with further optimization of training settings.

Task 2: LB Reranker Dataset The progressive perfor-mance improvement with increasing trainable parameters (from q=5 to q=30 ) reveals a clear parameter-performance tradeoff. Even with only q=20 , LoKI already achieves posi-tive performance gains in four metrics. As shown in Table 4, compared to all baseline methods including DoRA, PiSSA, and CorDA (Yang, Li et al. 2024), LoKI models of all con-figurations exhibit significantly less performance degrada-tion across all metrics. Interestingly, when q=30 , LoKI not only achieved the best averaged performance on the BEIR benchmark, but also showed the least degradation on general tasks. We attribute this phenomenon to our empirical learn-ing rate schedule: higher q values used proportionally lower learning rates (e.g., the learning rate for q=30 was 0.4 × that for q=10 ), which appears to provide a better balance be-tween task adaptation and knowledge preservation. Further exploration of this hyperparameter interplay is discussed in Appendix E.

# Ablation Studies

In this section, we evaluate (a) the effect of KVA in quantify-ing knowledge vectors’ contributions to general task perfor-mance, and (b) the necessity of the Layer-Balanced Strategy in LoKI.

Validation of KVA We use Llama3.1-8B-Instruct to val-idate if KVA can effectively attribute the contribution of knowledge vectors to general tasks. We compare two sup-pression strategies (i.e., zeroing the corresponding weights) with the base model using the same benchmarks in the Ex-periments section: • S-H/L : Suppresses the top q% knowledge vectors that appear most frequently with the highest /lowest attribu-tion scores (expected to significantly /minimally degrade performance if KVA correctly identifies critical vectors). As presented in Table 5a, when the top 1% of high-contribution vectors are suppressed, there is a signifi-cant performance decline, with an average degradation of 36.73%. In contrast, the performance gap between the two suppression strategies is 25.38% , which indicates signifi-cant differences and supports the accuracy of KVA.

Validation of Layer-Balanced Strategy Next, we test the necessity of the Layer-Balanced Strategy by fine-tuning Qwen2.5-0.5B-Instruct on the LB Rerank Dataset, using two globally imbalanced schemes. Equally, we compare these re-sults with models trained using LoKI on the 6 benchmarks mentioned above. • G-H/L : Globally set the top q% knowledge vectors across all layers that appear most frequently with the

highest /lowest attribution scores as trainable parameters. As shown in Table 5b, both imbalanced strategies fall short of LoKI, despite using the same quota, underscoring the im-portance of the Layer-Balanced Strategy. Additional BEIR benchmark results are included in Appendix G.

# Conclusions

We present LoKI, a parameter-efficient fine-tuning frame-work that achieves balanced adaptation between down-stream task performance and preservation of pre-trained knowledge in large language models. Our key insight stems from systematically analyzing the hierarchical knowledge storage mechanism in transformers and developing a layer-balanced parameter selection strategy guided by integrated gradient attribution. Through experiments on retrieval and tool-use tasks, we demonstrate that LoKI achieves competi-tive task adaptation while significantly reducing catastrophic forgetting compared to full parameter fine-tuning and preva-lent PEFT methods. Our experimental results demonstrate that integrating insights from mechanistic interpretability re-search with fine-tuning objectives is effective, highlighting the potential of this interdisciplinary direction. Model Overall Acc (%, ↑)Single Turn Acc Multi Turn (%, ↑)Hallucination Measurement Non-Live(%, ↑) Live(%, ↑) Relevance(%, ↑) Irrelevance(%, ↓)ToolACE †(r=16 ) 58.32 87.56 76.10 7.62 83.33 88.05 DoRA( r=16 ) 58.90 82.04 75.61 16.00 83.33 88.92 PiSSA( r=16 ) 53.97 80.56 72.68 5.62 61.11 92.19 LoKI( q=10 ) 56.76 80.81 70.46 16.50 83.33 82.21  

> LoKI( q=20 )58.16 81.02 73.26 17.75 83.33 85.73 LoKI( q=30 )58.93 82.71 75.70 16.25 83.33 87.65 LoKI*( q=30 , r =32 )57.16 81.96 71.66 15.75 83.33 84.01

Table 1: Performance comparison on the Berkeley Function Calling Leaderboard V3. ToolACE refers to the official model trained using LoRA. r denotes the rank. Models marked with † indicate that the corresponding performance metrics are sourced directly from the official BFCL documentation. An asterisk (*) denotes models where LoRA was applied during training.  

> Model #Params TriviaQA GSM8K Hellaswag WinoGrande HumanEval IFEval Avg( ↓)Llama3.1 (untuned) –65.77 84.46 73.85 62.98 68.29 79.76 –ToolACE( r=16 )42M 64.63 79.53 46.81 42.78 60.37 72.76 16.11% DoRA( r=16 )43M 63.97 82.64 70.78 57.85 65.24 73.47 4.92% PiSSA( r=16 )42M 52.50 51.40 37.98 13.18 15.85 55.95 48.93% LoKI( q=10 )188M 65.85 84.99 75.14 61.33 68.90 77.54 0.34% LoKI( q=20 )376M 65.60 84.31 73.60 60.62 67.68 78.18 0.93% LoKI( q=30 )563M 65.49 84.61 71.16 61.09 70.73 77.94 1.23% LoKI*( q=30 , r =32 )16M 64.30 83.47 70.78 62.51 70.12 78.21 1.26%

Table 2: Benchmark scores of models fine-tuned on ToolACE Function-Calling Dataset. Llama3.1 denotes Llama3.1-8B-Instruct. Avg denotes the average performance degradation percentage of each indicator compared to the original model in-dicator. We complete all the benchmarks on OpenCompass and report their results directly.  

> Model MAP@1 (%) Recall@1 (%) NDCG@1 (%) P@1 (%) Qwen2.5 -78.1 -78.1 -78.7 -77.5 DoRA( r=8 )-8.7 -8.7 -11.2 -10.4 PiSSA( r=8 )-9.6 -9.6 -9.7 -10.2 CorDA( r=8 )-3.3 -3.1 -2.6 -3.3 LoKI( q=5 )-3.1 -3.1 -2.2 -1.8 LoKI( q=10 )-2.3 -2.3 -0.3 -0.5 LoKI( q=20 )+0.2 +0.2 +0.7 +0.8 LoKI( q=30 )+1.0 +1.0 +2.1 +2.5

Table 3: Retrieval performance on BEIR benchmark. Results show the percentage difference relative to the full param-eter fine-tuning model (LB-Reranker) across standard re-trieval metrics, where positive values indicate superior per-formance. Bold represents the top performance score in each column; underline represents the runner-up. CorDA uses the KPM setting on the NQ-Open dataset.

# Acknowledgement

This work was supported in part by the National Natu-ral Science Foundation of China under Grants 52202496, 52442218, and U2433216; The Key Research and Devel-opment Project of Nantong City, China (Special Project for Prospective Technology Innovation, No. GZ2024001); and the Key Laboratory of Target Cognition and Application Technology (2023-CXPT-LC-005).

# References

Brown, T.; Mann, B.; et al. 2020. Language Models are Few-Shot Learners. In Advances in Neural Information Process-ing Systems , volume 33, 1877–1901. Chaudhry, A.; Ranzato, M.; et al. 2019. Efficient Lifelong Learning with A-GEM. In ICLR 2024 .Chen, M.; Tworek, J.; Jun, H.; et al. 2021. Evaluating Large Language Models Trained on Code. Clark, K.; Khandelwal, U.; et al. 2019. What Does BERT Look at? An Analysis of BERT’s Attention. In Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, BlackboxNLP@ACL 2019, Florence, Italy, August 1, 2019 , 276–286. Cobbe, K.; Kosaraju, V.; Bavarian, M.; et al. 2021. Train-ing Verifiers to Solve Math Word Problems. arXiv preprint arXiv:2110.14168 .Cohen, R.; Geva, M.; et al. 2023. Crawling The Inter-nal Knowledge-Base of Language Models. In Findings of the Association for Computational Linguistics: EACL 2023 ,1856–1869. Dubrovnik, Croatia. Dai, D.; Dong, L.; et al. 2022a. Knowledge Neurons in Pre-trained Transformers. In ACL , 8493–8502. Dai, D.; Dong, L.; et al. 2022b. Knowledge Neurons in Pre-trained Transformers. In ACL 2022 , 8493–8502. Dublin, Ire-land. Model #Params TriviaQA GSM8K HellaSwag WinoGrande HumanEval IFEval Avg( ↓)Qwen2.5 (untuned) – 24.37 39.65 30.98 44.20 26.83 33.39 –LB-Reranker 494M 4.65 2.65 0.00 0.00 0.00 23.20 84.13% DoRA( r=8 ) 4M 19.30 34.01 22.21 44.70 25.22 31.67 12.23% PiSSA( r=8 ) 4M 11.32 6.44 18.54 31.97 6.71 28.85 48.95% CorDA( r=8 ) 4M 4.64 2.43 0.00 0.00 0.00 23.20 84.22% LoKI( q=5 ) 5.1M 20.53 38.36 30.52 40.09 24.39 34.12 6.12% LoKI( q=10 ) 10.4M 20.53 37.98 21.53 47.04 24.39 33.39 8.86% LoKI( q=20 ) 20.9M 19.77 36.77 33.16 47.28 27.44 33.27 1.70% LoKI( q=30 ) 31.3M 20.10 37.60 38.39 43.96 26.83 32.25 0.46%

Table 4: Benchmark scores of models fine-tuned on the LB Reranker Dataset. LB-Reranker refers to the full parameter fine-tuning model released by the dataset provider. Qwen2.5 represents the untuned base model (Qwen2.5-0.5B-Instruct), which serves as the baseline for computing performance degradation.  

> (a) Model Avg( ↓)S-H 36.73% S-L 11.35% (b) Model Avg( ↓)LoKI 8.86% G-H 39.04% G-L 30.48%

Table 5: (a) Performance of two suppression strategies on benchmarks when q=1 . (b) Performance comparison of dif-ferent methods when q=10 . Detailed results are provided in Appendix G. de Masson d’Autume, C.; Ruder, S.; et al. 2019. Episodic Memory in Lifelong Language Learning. In NeurIPS 2019 ,13122–13131. DENG, D.; Chen, G.; et al. 2021. Flattening Sharpness for Dynamic Gradient Projection Memory Benefits Continual Learning. In Advances in Neural Information Processing Systems , volume 34, 18710–18721. Dubey, A.; et al. 2024. The Llama 3 Herd of Models. CoRR ,abs/2407.21783. Fang, J.; Jiang, H.; et al. 2025. AlphaEdit: Null-Space Con-strained Knowledge Editing for Language Models. In ICLR 2024 .Frankle, J.; and Carbin, M. 2019. The Lottery Ticket Hy-pothesis: Finding Sparse, Trainable Neural Networks. In

ICLR 2024 .Frantar, E.; Alistarh, D.; et al. 2023. SparseGPT: Mas-sive Language Models Can be Accurately Pruned in One-Shot. In ICML 2024 , volume 202 of Proceedings of Machine Learning Research , 10323–10337. French, R. M. 1993. Catastrophic Interference in Connec-tionist Networks: Can It Be Predicted, Can It Be Prevented? In NeurIPS , 1176–1177. Geva, M.; Bastings, J.; et al. 2023. Dissecting Recall of Fac-tual Associations in Auto-Regressive Language Models. In

EMNLP 2023 , 12216–12235. Singapore. Geva, M.; Caciularu, A.; et al. 2022. Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space. In EMNLP 2022 , 30–45. Abu Dhabi, United Arab Emirates. Geva, M.; Schuster, R.; et al. 2021. Transformer Feed-Forward Layers Are Key-Value Memories. In EMNLP 2021 ,5484–5495. Online and Punta Cana, Dominican Republic. Hao, Y.; Dong, L.; et al. 2021. Self-Attention Attribution: Interpreting Information Interactions Inside Transformer. In

AAAI 2021 , 12963–12971. Hase, P.; Bansal, M.; et al. 2023. Does Localization Inform Editing? Surprising Differences in Causality-Based Local-ization vs. Knowledge Editing in Language Models. In

NeurIPS 2023 .He, J.; Zhou, C.; et al. 2022. Towards a Unified View of Parameter-Efficient Transfer Learning. In ICLR 2024 .Hu, E. J.; Shen, Y.; et al. 2022. Lora: Low-rank adaptation of large language models. ICLR , 1(2): 3. Huang, J.; Cui, L.; et al. 2024. Mitigating Catastrophic For-getting in Large Language Models with Self-Synthesized Rehearsal. In ACL , 1416–1428. Joshi, M.; Choi, E.; et al. 2017. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Com-prehension. In ACL , 1601–1611. Katz, S.; Belinkov, Y.; et al. 2024. Backward Lens: Project-ing Language Model Gradients into the Vocabulary Space. In EMNLP 2024 , 2390–2422. Miami, Florida, USA. Kemker, R.; McClure, M.; et al. 2018a. Measuring catas-trophic forgetting in neural networks. In AAAI , 1. Kemker, R.; McClure, M.; et al. 2018b. Measuring Catas-trophic Forgetting in Neural Networks. In AAAI 2018 , 3390– 3398. Kirkpatrick, J.; Pascanu, R.; et al. 2016. Overcom-ing catastrophic forgetting in neural networks. CoRR ,abs/1612.00796. Kobayashi, G.; Kuribayashi, T.; et al. 2024. Analyzing Feed-Forward Blocks in Transformers through the Lens of Atten-tion Maps. In ICLR 2024 .Kotha, S.; Springer, J. M.; et al. 2024. Understanding Catas-trophic Forgetting in Language Models via Implicit Infer-ence. In ICLR 2024 .Lasby, M.; Golubeva, A.; et al. 2024. Dynamic Sparse Train-ing with Structured Sparsity. In ICLR 2024 .Li, H.; Ding, L.; et al. 2024. Revisiting Catastrophic Forget-ting in Large Language Model Tuning. In Findings of the Association for Computational Linguistics: EMNLP 2024 ,4297–4308. Miami, Florida, USA. Li, X.; Li, S.; et al. 2024. PMET: Precise Model Editing in a Transformer. In AAAI 2024 , 18564–18572. Li, Z.; and Hoiem, D. 2018. Learning without Forgetting.

IEEE Trans. Pattern Anal. Mach. Intell. , 40(12): 2935–2947. Liang, J.; Huang, W.; et al. 2025. LoRASculpt: Sculpting LoRA for Harmonizing General and Specialized Knowledge in Multimodal Large Language Models. In IEEE/CVF Con-ference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025 , 26170–26180. Lightblue. 2024. LB Reranker: A multilingual reranker model fine-tuned from Qwen2.5-0.5B-Instruct. https:// github.com/lightblue-tech/lb-reranker. Liu, S.; Wang, C.; et al. 2024. DoRA: Weight-Decomposed Low-Rank Adaptation. In ICML 2024 .Liu, W.; Huang, X.; et al. 2024. ToolACE: Winning the Points of LLM Function Calling. ArXiv:2409.00920 [cs]. Lopez-Paz, D.; and Ranzato, M. 2017. Gradient Episodic Memory for Continual Learning. In NeurIPS 2017 , 6467– 6476. Luo, Y.; Yang, Z.; et al. 2025. An Empirical Study of Catas-trophic Forgetting in Large Language Models During Con-tinual Fine-tuning. ArXiv:2308.08747 [cs]. Meng, F.; Wang, Z.; et al. 2024. PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models. In NeurIPS 2024 .Meng, K.; Bau, D.; et al. 2022. Locating and Editing Factual Associations in GPT. In Advances in Neural Information Processing Systems , volume 35, 17359–17372. Meng, K.; Sharma, A. S.; et al. 2023a. Mass-Editing Mem-ory in a Transformer. In ICLR 2024 .Meng, K.; Sharma, A. S.; et al. 2023b. Mass-Editing Mem-ory in a Transformer. ArXiv:2210.07229 [cs]. Michel, P.; Levy, O.; et al. 2019. Are Sixteen Heads Really Better than One? In NeurIPS 2019 , 14014–14024. Petroni, F.; Rockt¨ aschel, T.; et al. 2019. Language Models as Knowledge Bases? In EMNLP-IJCNLP 2019 , 2463–2473. Hong Kong, China. Prottasha, N. J.; Chowdhury, U. R.; et al. 2025. PEFT A2Z: Parameter-Efficient Fine-Tuning Survey for Large Language and Vision Models. ArXiv:2504.14117 [cs]. Qwen; Yang, A.; et al. 2025. Qwen2.5 Technical Report. ArXiv:2412.15115 [cs]. Radford, A.; Narasimhan, K.; et al. 2018. Improving lan-guage understanding by generative pre-training. Sakaguchi, K.; Bras, R. L.; et al. 2020. WinoGrande: An Adversarial Winograd Schema Challenge at Scale. In AAAI 2020 , 8732–8740. Sanh, V.; Wolf, T.; et al. 2020a. Movement Pruning: Adap-tive Sparsity by Fine-Tuning. In NeurIPS 2020 .Sanh, V.; Wolf, T.; et al. 2020b. Movement Pruning: Adap-tive Sparsity by Fine-Tuning. In Advances in Neural Infor-mation Processing Systems , volume 33, 20378–20389. Sun, Q.; Pickett, M.; et al. 2025a. Transformer Layers as Painters. In AAAI-25 , 25219–25227. Sun, Q.; Pickett, M.; et al. 2025b. Transformer layers as painters. In AAAI , 24, 25219–25227. Sundararajan, M.; Taly, A.; et al. 2017. Axiomatic Attri-bution for Deep Networks. In ICML 2024 , volume 70 of

Proceedings of Machine Learning Research , 3319–3328. Tan, C.; Zhang, G.; et al. 2024. Massive Editing for Large Language Models via Meta Learning. In ICLR 2024 .Tenney, I.; Das, D.; et al. 2019. BERT Rediscovers the Clas-sical NLP Pipeline. In ACL 2019 , 4593–4601. Florence, Italy. Vaswani, A.; Shazeer, N.; et al. 2017. Attention is All you Need. In Advances in Neural Information Processing Sys-tems , volume 30. Vig, J.; and Belinkov, Y. 2019. Analyzing the Structure of Attention in a Transformer Language Model. In Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, BlackboxNLP@ACL 2019, Florence, Italy, August 1, 2019 , 63–76. Voita, E.; Talbot, D.; et al. 2019. Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned. In ACL 2019 , 5797–5808. Florence, Italy. Wallat, J.; Singh, J.; et al. 2020. BERTnesia: Investigating the capture and forgetting of knowledge in BERT. In Black-boxNLP 2019 , 174–183. Online. Wang, X.; Chen, T.; et al. 2023. Orthogonal Subspace Learn-ing for Language Model Continual Learning. In Findings of EMNLP , 10658–10671. Xin, Y.; Yang, J.; et al. 2025. Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey. ArXiv:2402.02242 [cs]. Yang, Y.; Li, X.; et al. 2024. CorDA: Context-Oriented Decomposition Adaptation of Large Language Models for Task-Aware Parameter-Efficient Fine-tuning. In NeurIPS 2024 .Zellers, R.; Holtzman, A.; et al. 2019. HellaSwag: Can a Machine Really Finish Your Sentence? In ACL , 4791–4800. Zhang, J.; You, J.; et al. 2025. LoRI: Reducing Cross-Task Interference in Multi-Task Low-Rank Adaptation. CoRR ,abs/2504.07448. Zhang, Q.; Chen, M.; et al. 2023. Adaptive Budget Alloca-tion for Parameter-Efficient Fine-Tuning. In ICLR 2024 .Zhang, Z.; Lin, Y.; et al. 2022. MoEfication: Transformer Feed-forward Layers are Mixtures of Experts. In Findings of ACL , 877–890. Zhou, J.; Lu, T.; et al. 2023. Instruction-Following Evalua-tion for Large Language Models. ArXiv:2311.07911 [cs]. Zhu, D.; Sun, Z.; et al. 2024. Model Tailor: Mitigat-ing Catastrophic Forgetting in Multi-modal Large Language Models. In ICML 2024 .
