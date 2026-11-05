# Golden questions for review

## nested_learning

| id | split | type | question | from |
|---|---|---|---|---|
| q01 | test | single | What is the purpose of separating the storage of Routing Keys and Content KVs in the Memory Sparse Attention method, and how does this design choice impact the model's capacity requirements? | MSA: Memory Sparse Attention for Efficient En |
| q02 | dev | single | What is a limitation of inter-layer hybrid architectures in long-context modeling? | AllMem: A Memory-centric Recipe for Efficient |
| q03 | dev | single | How do recurrent models differ from traditional MLP blocks according to the nested learning perspective? | Nested Learning: The Illusion of Deep Learnin |
| q04 | test | single | How does the memory unit M_t utilize the sequence of token embeddings to predict the next token in an auto-regressive language model? | Beyond Test-Time Training: Learning to Reason |
| q05 | dev | single | What is the trade-off associated with logarithmic segmentation in memory caching, and how does it affect recall performance? | Memory Caching: RNNs with Growing Memory |
| q06 | dev | single | How does the feature refinement mechanism in CRAD-HOPE combine different similarity measures to improve anomaly detection? | CRAD-HOPE: Brain-inspired Nested Learning Fra |
| q07 | dev | single | How does the Dual UT method improve learning dynamics compared to single-stream tying in autoregressive models? | Tiny Autoregressive Recursive Models |
| q08 | test | single | What is the role of the geometry-aware optimizer ARROW in addressing plasticity loss in Vision Transformers? | Vision Transformers That Never Stop Learning |
| q09 | spare | single | What is the purpose of incorporating sparsity in the memory weights of the EGD-MEMORY model? | HYDRA: Dual Exponentiated Memory for Multivar |
| q10 | dev | single | What is the significance of the state transition matrices At and Bt in the context of DeltaNet's expressivity? | Fast Weight Programming and Linear Transforme |
| q11 | test | single | What is the role of the meta-optimization process in the Dynamic Nested Hierarchy model, and how does it address the limitations of long-context reasoning? | Dynamic Nested Hierarchies: Self-Evolution in |
| q12 | test | single | What limitation does the RAG system have in terms of updating facts, and how does the KO method address this limitation? | Attention Is Not Retention: The Orthogonality |
| q13 | spare | single | What is the key mechanism proposed for improving language modeling performance with longer context while maintaining constant cost per token? | End-to-End Test-Time Training for Long Contex |
| q14 | dev | single | What is the purpose of the Observability Score Ratio (OSR) in evaluating agent outputs, and how is it calculated? | Prompt Injection Mitigation with Agentic AI,  |
| q15 | test | single | What limitation is identified regarding Learning-to-Learn (L2L) methods in terms of task scope and scale? | Learning-to-Learn at Scale: Promises and Chal |
| q16 | test | single | How does the Memory Sparse Attention method reduce memory footprint and retrieval complexity during the processing of documents? | MSA: Memory Sparse Attention for Efficient En |
| q17 | dev | single | What advantages does the ALLMEM architecture provide for long-sequence modeling compared to the full-attention mechanism? | AllMem: A Memory-centric Recipe for Efficient |
| q18 | test | single | How does Delta Gradient Descent (DGD) differ from traditional gradient descent in terms of its update mechanism? | Nested Learning: The Illusion of Deep Learnin |
| q20 | dev | single | What limitation does the Residual Memory formulation have in relation to cached memories, and how does the introduction of input-dependent gating address this limitation? | Memory Caching: RNNs with Growing Memory |
| q21 | test | single | How does the CRAD-HOPE framework prevent catastrophic forgetting during the learning process? | CRAD-HOPE: Brain-inspired Nested Learning Fra |
| q22 | dev | single | What limitation does the autoregressive TRM exhibit in relation to the tasks of Copy and Reverse, and what does this suggest about its learning capability? | Tiny Autoregressive Recursive Models |
| q23 | test | single | What limitation is observed in Vision Transformers during long-term continual learning, despite their structural advantages over MLPs? | Vision Transformers That Never Stop Learning |
| q24 | dev | single | What challenges do Transformer-based architectures face when modeling multivariate time series data? | HYDRA: Dual Exponentiated Memory for Multivar |
| q25 | test | single | How does the concept of Fast Weight Programming (FWP) relate to local online learning and metalearning in the context of neural networks? | Fast Weight Programming and Linear Transforme |
| q26 | spare | single | What advantage does the Dynamic Nested Hierarchies (DNH) method have over static Nested Learning (NL) in terms of adaptation efficiency, particularly in non-stationary settings? | Dynamic Nested Hierarchies: Self-Evolution in |
| q27 | test | single | What advantages does the Knowledge Objects system provide for handling patient data compared to pure neural systems or pure RAG systems? | Attention Is Not Retention: The Orthogonality |
| q28 | dev | single | How does the TTT-KVB method update its weights during training, and what role do the outer-loop parameters play in this process? | End-to-End Test-Time Training for Long Contex |
| q30 | dev | cross | How do the optimization objectives differ between the dual exponentiated memory approach and the memory caching framework in their respective methods for learning mappings between keys and values? | HYDRA: Dual Exponentiated Memory for Multivar; Memory Caching: RNNs with Growing Memory |
| q31 | test | cross | How do the Memory Sparse Attention (MSA) and Memory Caching (MC) approaches differ in their handling of memory capacity and computational complexity for long-context tasks? | MSA: Memory Sparse Attention for Efficient En; Memory Caching: RNNs with Growing Memory |
| q33 | dev | cross | How do self-evolving DNH models differ from traditional deep learning architectures in terms of adaptability to changing data distributions? | Dynamic Nested Hierarchies: Self-Evolution in; Nested Learning: The Illusion of Deep Learnin |
| q34 | test | cross | How do the approaches to addressing plasticity loss in continual learning differ between the Vision Transformer method and the Tiny Autoregressive Recursive Model? | Vision Transformers That Never Stop Learning; Tiny Autoregressive Recursive Models |
| q35 | dev | cross | How do the memory processing mechanisms of AllMem and MSA differ in terms of computational complexity during inference? | AllMem: A Memory-centric Recipe for Efficient; MSA: Memory Sparse Attention for Efficient En |
| q36 | test | cross | How do the concepts of fast weights in Fast Weight Programming (FWP) and test-time training (TTT) relate in terms of their approach to updating model parameters? | Fast Weight Programming and Linear Transforme; End-to-End Test-Time Training for Long Contex |

## titans

| id | split | type | question | from |
|---|---|---|---|---|
| q01 | dev | single | What is the relationship between sparse attention mechanisms and kernel regression as established in the research? | Sparse Attention as Compact Kernel Regression |
| q02 | dev | single | How does test-time training enhance the expressiveness of memory in the context of recurrent models? | ParaTTT: Parallelizing Test-Time Training Lay |
| q03 | test | single | How does MemGPT manage different types of memory to optimize performance? | Memory-Augmented Transformers: A Systematic R |
| q04 | dev | single | What role do intermediate layers play in the reasoning performance of models like GPT-2? | A Survey on Latent Reasoning |
| q05 | test | single | How does the TARSE method improve reasoning accuracy compared to other configurations in the evaluation settings? | TARSE: Test-Time Adaptation via Retrieval of  |
| q06 | test | single | What is the significance of the data-dependent decay factor in the context of Test-Time Training, and how does it relate to in-context learning performance? | Test-Time Training with KV Binding Is Secretl |
| q07 | dev | single | What are the three components of the Titans architecture and their respective roles in processing data? | Titans: Learning to Memorize at Test Time |
| q08 | dev | single | How does CommVQ address the memory bottleneck associated with KV cache in autoregressive Transformers? | Memory for Large Language Models |
| q09 | dev | single | How does E2-TTT improve upon traditional test-time training methods in terms of efficiency and expressivity? | Rethinking Expressivity and Efficiency in Tes |
| q10 | test | single | What is the purpose of Hierarchical Voting in the context of augmented inference, and how does it compare to Flattened Voting based on the results presented? | The Surprising Effectiveness of Test-Time Tra |
| q11 | dev | single | What is the main computational limitation of the softmax attention mechanism in sequence models, and how does it affect the generation of long sequences? | Test-time regression: a unifying framework fo |
| q12 | spare | single | What is the impact of using a uniform distribution for training horizons compared to a fixed horizon in terms of performance and computational cost? | Beyond Test-Time Memory: State-Space Optimal  |
| q13 | test | single | What impact does varying the chunk size have on the performance and memory usage of the ChangeTitans model? | Towards Remote Sensing Change Detection with  |
| q14 | spare | single | What is the purpose of including persistent memory in the PriorEye model? | PriorEye: Geospatial Visual Priors for End-to |
| q15 | dev | single | How does the ReLUmax method avoid the 0/0 degeneracy found in normalized ReLU, and what implications does this have for its performance? | Sparse Attention as Compact Kernel Regression |
| q16 | test | single | How does Test-time Training (TTT) enhance the expressivity of recurrent memory in models while maintaining parallel evaluation? | ParaTTT: Parallelizing Test-Time Training Lay |
| q17 | spare | single | What limitation is identified regarding the SR-TTT method in relation to early-context needles and relative-surprisal signals? | SR-TTT Does Not Learn Retrieval: A Correction |
| q18 | test | single | How do hybrid memory systems enhance the performance of Transformers in complex reasoning applications? | Memory-Augmented Transformers: A Systematic R |
| q19 | test | single | How do Masked Diffusion Models (MDMs) utilize bidirectional context in their token prediction process? | A Survey on Latent Reasoning |
| q21 | test | single | What does the relationship between inner-loop loss and downstream performance suggest about the effectiveness of memorization in test-time training (TTT)? | Test-Time Training with KV Binding Is Secretl |
| q22 | test | single | What is a limitation of Transformers in sequence modeling, and how does Titans address this limitation? | Titans: Learning to Memorize at Test Time |
| q23 | dev | single | How does the dual-memory regime in memory architectures facilitate selective recall without increasing computational cost? | Memory for Large Language Models |
| q24 | dev | single | How does the token-wise state recurrence method retain per-token momentum and multiplicative decay in its updates? | Rethinking Expressivity and Efficiency in Tes |
| q25 | dev | single | How does test-time training (TTT) enhance the performance of a model in few-shot learning tasks? | The Surprising Effectiveness of Test-Time Tra |
| q26 | test | single | How does DeltaNet address the challenge of forgetting past information in sequence models? | Test-time regression: a unifying framework fo |
| q28 | test | single | What advantage does MSCANet provide in handling SAR change detection compared to existing methods? | Towards Remote Sensing Change Detection with  |
| q29 | dev | single | How does the method for querying geospatial visual priors ensure relevance to the current driving context? | PriorEye: Geospatial Visual Priors for End-to |
| q30 | spare | cross | How do the concepts of vertical and horizontal recurrence in Latent CoT implementations relate to the improvements in inference throughput achieved by the parallel formulation in test-time training with KV binding? | A Survey on Latent Reasoning; Test-Time Training with KV Binding Is Secretl |
| q31 | dev | cross | How do the approaches to surprise and memory management differ between the methods described in SR-TTT and Titans, particularly in terms of handling context and relevance of tokens? | SR-TTT Does Not Learn Retrieval: A Correction; Titans: Learning to Memorize at Test Time |
| q32 | test | cross | How do the kernel regression approaches in Memory Mosaics and local polynomial regression differ in their treatment of locality and the role of the query and key in the regression process? | Sparse Attention as Compact Kernel Regression; Test-time regression: a unifying framework fo |
| q33 | test | cross | How does the VTitans framework for remote sensing change detection relate to the design tradeoffs in hybrid memory architectures discussed in the context of large language models? | Towards Remote Sensing Change Detection with ; Memory for Large Language Models |
| q34 | test | cross | How does the performance of the ParaTTT model compare to the linear attention approach in terms of perplexity and iteration counts? | ParaTTT: Parallelizing Test-Time Training Lay; Test-Time Training with KV Binding Is Secretl |
| q35 | dev | cross | How do the concepts of explicit memory management and the architecture of memory systems in memory-augmented Transformers relate to each other in terms of their implications for model complexity and interpretability? | Memory for Large Language Models; Memory-Augmented Transformers: A Systematic R |
| q36 | dev | cross | How does the Longhorn sequence layer's adaptive step size mechanism compare to the step size approach used in the Gated DeltaNet readout in terms of their underlying objectives and implementation? | Test-time regression: a unifying framework fo; ParaTTT: Parallelizing Test-Time Training Lay |

## ttt_rnn

| id | split | type | question | from |
|---|---|---|---|---|
| q01 | dev | single | How does the Modular TTT method enable the composition of a full graph-level TTT computation without needing to derive a new global update rule for each variant? | Modular TTT: Rethinking Test-Time Training as |
| q02 | test | single | How does the fast weight update mechanism in the tttLRM model help in online 3D Gaussian reconstruction? | tttLRM: Test-Time Training for Long Context a |
| q03 | dev | single | What is the significance of applying LayerNorm before a squared-error loss in the context of TTT systems? | Error Correction in Trained TTT Layers |
| q05 | dev | single | How does the adaptive correction mechanism in the PRISM model ensure that updates are suppressed in saturated memory regions? | PRISM: Parallel Residual Iterative Sequence M |
| q06 | test | single | What limitation does TTT3R face when processing long sequences, and how does it attempt to address this issue? | Scal3R: Scalable Test-Time Training for Large |
| q07 | dev | single | What challenge does the end-to-end chunk-wise processing method introduce when managing long sequences for 3D reconstruction? | LoGeR: Long-Context Geometric Reconstruction  |
| q08 | test | single | What limitation of the transformer model is highlighted in relation to state tracking, and what alternative approach is suggested to address this limitation? | The Topological Trouble With Transformers |
| q09 | test | single | What limitation do full-attention methods like VGGT and StreamVGGT face compared to online methods in the context of camera pose estimation? | TTT3R: 3D Reconstruction as Test-Time Trainin |
| q10 | test | single | What is the significance of conflict-aware knowledge integration in large language models, and what challenge does it address? | Open Problems and a Hypothetical Path Forward |
| q11 | dev | single | What is the purpose of Test-Time Adaptation (TTA) in continual learning? | Continual Learning in Transition |
| q12 | dev | single | What is the open-loop bottleneck in current transformer models, and what are its implications for long-form generation and reasoning tasks? | Closed-Loop Transformers: Autoregressive Mode |
| q13 | test | single | What is the significance of adapting 'frame consistency' to 'temporal consistency' in the context of TTT-MLP's performance? | One-Minute Video Generation with Test-Time Tr |
| q14 | dev | single | What limitation is observed when using standard backbones like bidirectional and autoregressive Transformers for tracking hidden states in video models? | Can Video World Models Track Unobserved World |
| q15 | test | single | How does the TTCD method encourage a model to store useful distant information for future predictions? | Learning What to Remember: Test-Time Training |
| q16 | test | single | How does Modular TTT address the challenge of determining the contributions of different design choices in test-time training? | Modular TTT: Rethinking Test-Time Training as |
| q17 | dev | single | How does pretrained initialization impact the training of the GS model in terms of convergence speed and final quality? | tttLRM: Test-Time Training for Long Context a |
| q18 | dev | single | How does the classical delta rule facilitate the updating of associations in a linear fast-weight memory when a new observation is made? | Error Correction in Trained TTT Layers |
| q20 | spare | single | How does PRISM achieve both Rank-L and parallel scan compatibility in its optimization process? | PRISM: Parallel Residual Iterative Sequence M |
| q21 | test | single | What is the purpose of introducing fast weights in Test-Time Training, and how do they address the limitations of traditional recurrent neural networks? | Scal3R: Scalable Test-Time Training for Large |
| q22 | test | single | How does the progressive curriculum strategy improve the training of the LoGeR model? | LoGeR: Long-Context Geometric Reconstruction  |
| q24 | test | single | What is the advantage of using causal attention in the StreamVGGT method for processing streaming input, and what limitation does it share with full attention? | TTT3R: 3D Reconstruction as Test-Time Trainin |
| q25 | dev | single | What are the three critical open problems that limit the capabilities of Large Language Models according to the discussed knowledge paradigms? | Open Problems and a Hypothetical Path Forward |
| q26 | spare | single | How does the SKILL0 method accumulate and internalize skills in its learning process? | Continual Learning in Transition |
| q27 | test | single | What is the significance of using the binary cumulative parity task for evaluating autoregressive models in the context of Closed-Loop Transformers? | Closed-Loop Transformers: Autoregressive Mode |
| q28 | dev | single | How does the TATS method improve video generation compared to traditional approaches? | One-Minute Video Generation with Test-Time Tr |
| q29 | dev | single | What limitation do standard backbones face when trained on action-conditioned video tasks like the Shell Game, and how does this affect their ability to generalize to longer swap chains? | Can Video World Models Track Unobserved World |
| q30 | test | cross | How do TTT3R and Scal3R compare in terms of accuracy and runtime efficiency for 3D reconstruction on long sequences? | TTT3R: 3D Reconstruction as Test-Time Trainin; Scal3R: Scalable Test-Time Training for Large |
| q31 | dev | cross | How do the Long-LRM and TTT3R approaches address the challenges of scalability and memory usage in 3D reconstruction from multiple input views? | tttLRM: Test-Time Training for Long Context a; TTT3R: 3D Reconstruction as Test-Time Trainin |
| q33 | dev | cross | How does the computational complexity of the PRISM model compare to that of the TTT model when processing input sequences, and what implications does this have for their respective throughput performance? | PRISM: Parallel Residual Iterative Sequence M; TTT3R: 3D Reconstruction as Test-Time Trainin |
| q34 | test | cross | How do the training throughput and latency characteristics of Modular TTT compare to those of the RNN models discussed in the context of learning at test time, particularly in relation to the TTT-Linear and TTT-MLP implementations? | Modular TTT: Rethinking Test-Time Training as; Learning to (Learn at Test Time): RNNs with E |
| q35 | test | cross | How do the probabilistic knowledge acquisition methods in language models relate to the iterative reasoning process proposed in Closed-Loop Transformers? | Open Problems and a Hypothetical Path Forward; Closed-Loop Transformers: Autoregressive Mode |
| q36 | dev | cross | How do the closed-loop principle in Equilibrium Transformers and the proposed hidden state model in future architectures address the challenges of multi-step reasoning and knowledge accommodation in language models? | Closed-Loop Transformers: Autoregressive Mode; Open Problems and a Hypothetical Path Forward |

## hrm

| id | split | type | question | from |
|---|---|---|---|---|
| q01 | dev | single | What is the purpose of the InfiLoop method in looped Transformers, and how does it address the issue of latent overthinking? | Scaling to Tens of Thousands of Test-Time Ite |
| q02 | test | single | What limitation does the use of truncated backpropagation through time introduce in the training of Neural Cellular Automata, and what future research directions are suggested to address this limitation? | Reasoning with Neural Cellular Automata |
| q03 | test | single | How does the hierarchical reasoning model utilize temporal separation to enhance computational efficiency? | Hierarchical Reasoning Model |
| q05 | dev | single | What is the significance of using matched controls when testing the claim that a shallow block looped can replace stacked layers in recursive models? | Evaluating Tiny Recursive Models Across Train |
| q07 | dev | single | What is the significance of the scaling law L(t, N, L, P) in the context of model performance for in-context learning tasks, particularly regarding the effects of depth and width? | Theory of Scaling Laws for In-Context Regress |
| q08 | dev | single | What is the significance of the contraction mapping property in the Contraction Mapping Model (CMM) and how does it contribute to the model's convergence to a unique fixed point? | Dynamical Systems Theory Behind a Hierarchica |
| q09 | test | single | What is the significance of using an ensemble of checkpoints from the training phase in improving model accuracy, and what unexpected result was observed from this approach? | Are Your Reasoning Models Reasoning or Guessi |
| q10 | test | single | What is the significance of the condition ρ(Mext) < 1 in the context of attracting fixed points in the Mext model? | Stability and Generalization in Looped Transf |
| q11 | test | single | What is the purpose of the boundedness theorem in Fixed-Point Reasoners, and how does it relate to the challenges faced by Euclidean formulations? | Spherical Fixed-Point Reasoners: Adaptive Loo |
| q13 | test | single | How do looped reasoning models increase computational depth without increasing the parameter count? | Looped Reasoning Finishes Earlier Than You Th |
| q14 | dev | single | How does the width scaling method in the Probabilistic Tiny Recursive Model (PTRM) enhance performance during test-time computation? | Probabilistic Tiny Recursive Model |
| q15 | test | single | How does the dual phase–frequency state design in Winfree Oscillatory Neural Networks contribute to its performance in reasoning tasks? | Winfree Oscillatory Neural Network |
| q16 | dev | single | What advantage does the InfiLoop method provide over the carry-last model in terms of maintaining solution stability during deep reasoning processes? | Scaling to Tens of Thousands of Test-Time Ite |
| q17 | dev | single | How do Neural Cellular Automata (NCAs) demonstrate generalization capabilities to harder, out-of-distribution tasks? | Reasoning with Neural Cellular Automata |
| q18 | dev | single | How does the Hierarchical Reasoning Model (HRM) achieve superior performance on inductive benchmarks compared to traditional chain-of-thought (CoT) models? | Hierarchical Reasoning Model |
| q20 | test | single | What is the rationale behind selecting the best-validation epoch post hoc for the TRM-AR and vanilla-2L models, and how does this approach address potential limitations of early stopping methods? | Evaluating Tiny Recursive Models Across Train |
| q21 | dev | single | What limitation is identified regarding the computation performed by deep equilibrium models, and how is this limitation demonstrated through testing? | The Equilibrium Is the Initialization: Lazy I |
| q22 | test | single | What does the analysis of the model suggest about the necessity of depth in in-context learning when training on fixed covariance structures? | Theory of Scaling Laws for In-Context Regress |
| q23 | dev | single | How does the application of the NSDE framework contribute to the recovery of predictive capability in the CMMNSDE model despite its small parameter space? | Dynamical Systems Theory Behind a Hierarchica |
| q24 | test | single | What limitation does the hierarchical reasoning model (HRM) face when solving extremely simple puzzles, and what solution is proposed to address this limitation? | Are Your Reasoning Models Reasoning or Guessi |
| q25 | dev | single | What is the purpose of the KL-divergence regularization term introduced by Banino et al. in their method, and how does it differ from the goals of other models that reduce computation at inference time? | Stability and Generalization in Looped Transf |
| q26 | test | single | How does the design choice of using tied residual coefficients in the Fixed-Point Reasoning Model (FPRM) contribute to the stability of the recurrent carry path? | Spherical Fixed-Point Reasoners: Adaptive Loo |
| q27 | spare | single | What is a limitation of C-voting when applied to the HRM model, and how does it affect performance? | C-voting: Confidence-Based Test-Time Voting w |
| q28 | test | single | How does the removal of the oscillatory coordinate affect the settling time in the context of the FPRM model? | Looped Reasoning Finishes Earlier Than You Th |
| q29 | dev | single | How does the Probabilistic TRM (PTRM) improve accuracy during inference compared to standard methods? | Probabilistic Tiny Recursive Model |
| q30 | dev | cross | How do the state update mechanisms differ between the Spherical Fixed-Point Reasoners model and the Hierarchical Reasoning Model? | Spherical Fixed-Point Reasoners: Adaptive Loo; Hierarchical Reasoning Model |
| q31 | dev | cross | How does the performance of the ItrSA++ model with C-voting compare to the FPRM model in terms of solving Sudoku-Extreme, and what are the implications of their respective parameter counts? | Looped Reasoning Finishes Earlier Than You Th; C-voting: Confidence-Based Test-Time Voting w |
| q32 | dev | cross | How do the energy-controlled refinement method in Draft Refinement and the adaptive updates in Loop-Native Attention Residuals differ in their approach to improving solution quality during iterative processes? | Breaking Diffusion: Parallel Reasoning with D; Scaling to Tens of Thousands of Test-Time Ite |
| q33 | test | cross | How does the concept of equilibrium points in the Dynamical Systems Theory relate to the training dynamics of the Hierarchical Reasoning Model, particularly in terms of the stability of the hidden states? | Dynamical Systems Theory Behind a Hierarchica; Hierarchical Reasoning Model |
| q34 | test | cross | How does the performance of the Probabilistic Tiny Recursive Model (PTRM) compare to the Hierarchical Reasoning Model (HRM) on Sudoku-Extreme, and what implications does this have for their respective approaches to solving complex reasoning tasks? | Probabilistic Tiny Recursive Model; Hierarchical Reasoning Model |
| q35 | test | cross | How do the approaches of the Winfree Oscillatory Neural Network (WONN) and Neural Cellular Automata (NCA) differ in terms of task complexity and model scalability? | Winfree Oscillatory Neural Network; Reasoning with Neural Cellular Automata |
| q36 | spare | cross | How do the stability conditions of the dynamics in Looped Transformers relate to the weight renormalization and trajectory stabilization methods used in the Hierarchical Reasoning Model? | Stability and Generalization in Looped Transf; Dynamical Systems Theory Behind a Hierarchica |
