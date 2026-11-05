# Same question, different settings

## Stage 1: judge on nested_learning

**Question:** How do the optimization objectives differ between the dual exponentiated memory approach and the memory caching framework in their respective methods for learning mappings between keys and values?

### gpt-4o-mini: faithfulness no claims made

The provided context does not contain enough information to answer the question regarding the differences in optimization objectives between the dual exponentiated memory approach and the memory caching framework.

### gpt-5-nano: faithfulness 0.00

The provided context does not contain enough information to answer the question regarding the differences in optimization objectives between the dual exponentiated memory approach and the memory caching framework.

### gemini-3.5-flash-lite: faithfulness no claims made

The provided context does not contain enough information to answer the question regarding the differences in optimization objectives between the dual exponentiated memory approach and the memory caching framework.

### gpt-oss-20b: faithfulness 1.00

The provided context does not contain enough information to answer the question regarding the differences in optimization objectives between the dual exponentiated memory approach and the memory caching framework.

### gpt-4o-mini-rubric (chosen): faithfulness no claims made

The provided context does not contain enough information to answer the question regarding the differences in optimization objectives between the dual exponentiated memory approach and the memory caching framework.

## Stage 1: chunking on nested_learning

**Question:** How do the memory processing mechanisms of AllMem and MSA differ in terms of computational complexity during inference?

### fixed-512-50: faithfulness no claims made

The provided context does not contain information about the memory processing mechanisms of AllMem, nor does it provide a comparison of its computational complexity during inference with that of MSA. Therefore, I cannot answer the question based on the given context.

### recursive-400-80: faithfulness no claims made

The provided context does not contain enough information to directly compare the computational complexity of the memory processing mechanisms of AllMem and MSA during inference. While MSA's inference complexity is mentioned, the specific computational complexity details for AllMem during inference are not provided. Therefore, a comparison cannot be made.

### semantic-p70: faithfulness 1.00

The memory processing mechanisms of AllMem and MSA differ in terms of computational complexity during inference as follows:

- **AllMem**: The ALLMEM architecture achieves a significant reduction in computational cost and cache storage requirements compared to the full-attention mechanism. Specifically, when the context length extends to 128k, the computational cost and cache storage requirement of the ALLMEM model are reduced to approximately 1/9 of those of the full-attention mechanism. The cache capacity of the ALLMEM model remains constant regardless of sequence length, which contrasts with the full-attention layer that grows linearly with sequence length.

- **MSA**: The Memory Sparse Attention (MSA) achieves linear computational complexity with respect to memory size L during inference, specifically O(L) efficiency. This allows MSA to maintain consistent performance while processing large memory banks, with minimal degradation in quality across varying context lengths.

In summary, ALLMEM reduces computational costs significantly compared to full attention, while MSA maintains linear complexity with respect to memory size, allowing for efficient processing during inference.

### recursive-800-120 (chosen): faithfulness no claims made

The context does not provide enough information to directly compare the memory processing mechanisms of AllMem and MSA in terms of computational complexity during inference. While MSA achieves linear complexity in both training and inference, the specific computational complexity details for AllMem during inference are not provided in the context. Therefore, a direct comparison cannot be made.

## Stage 1: embedding on nested_learning

**Question:** How do the memory processing mechanisms of AllMem and MSA differ in terms of computational complexity during inference?

### all-MiniLM-L6-v2: faithfulness 1.00

The context does not provide enough information to explicitly compare the memory processing mechanisms of AllMem and MSA in terms of computational complexity during inference. While MSA's inference complexity involves a one-time offline pre-processing step with O(LG) complexity, the context does not detail the inference complexity of AllMem for a direct comparison.

### text-embedding-3-small (chosen): faithfulness no claims made

The context does not provide enough information to directly compare the memory processing mechanisms of AllMem and MSA in terms of computational complexity during inference. While MSA achieves linear complexity in both training and inference, the specific computational complexity details for AllMem during inference are not provided in the context. Therefore, a direct comparison cannot be made.

## Stage 2: retrieval_mode on titans

**Question:** How does the dual-memory regime in memory architectures facilitate selective recall without increasing computational cost?

### semantic: faithfulness no claims made

The provided context does not contain enough information to explicitly explain how the dual-memory regime in memory architectures facilitates selective recall without increasing computational cost.

### hybrid: faithfulness no claims made

The provided context does not contain enough information to explicitly explain how the dual-memory regime in memory architectures facilitates selective recall without increasing computational cost.

### hybrid-rerank: faithfulness no claims made

The provided context does not contain enough information to specifically explain how the dual-memory regime in memory architectures facilitates selective recall without increasing computational cost.

### keyword (chosen): faithfulness 1.00

The dual-memory regime in memory architectures facilitates selective recall without increasing computational cost by incorporating scalable key-based retrieval mechanisms within the model's early layers. This allows explicit memory storage to coexist with dense Transformer computation, enabling the model to route tokens through sparse memory slots effectively. As a result, the architecture can achieve selective recall while avoiding a uniform increase in computational demands.

## Stage 2: passages_used on titans

**Question:** How do the approaches to surprise and memory management differ between the methods described in SR-TTT and Titans, particularly in terms of handling context and relevance of tokens?

### top-3: faithfulness no claims made

The provided context does not contain enough information to answer the question regarding the differences in approaches to surprise and memory management between the methods described in SR-TTT and Titans, particularly in terms of handling context and relevance of tokens.

### top-5: faithfulness 0.91

The approaches to surprise and memory management differ between SR-TTT and Titans primarily in how they handle context and the relevance of tokens. Titans utilize a gradient-based "surprise" metric that gates memorization into a neural memory, where the surprise of previous tokens can influence the surprise of the next token, contingent on the relevance of those tokens within the same context. This is managed through a data-dependent decay parameter (𝜂𝑡), which can either ignore the last surprise or fully incorporate it based on the context's relevance.

In contrast, SR-TTT's routing signal is closely related to reconstruction loss and its gradient magnitude, which is position-biased and requires a burn-in period. This suggests that SR-TTT's approach may not be as flexible in adapting to changes in context as Titans, which can dynamically adjust the influence of past surprises based on the relevance of tokens. Thus, while both methods aim to enhance memory and surprise management, Titans offer a more nuanced mechanism for incorporating context and token relevance into their memory management strategy.

### top-8 (chosen): faithfulness 0.82

The approaches to surprise and memory management between SR-TTT and Titans differ primarily in how they handle context and the relevance of tokens. 

Titans utilize a gradient-based "surprise" mechanism to gate memorization into a neural memory, where the surprise metric is influenced by the relevance of previous tokens within the same context. Specifically, Titans employ a data-dependent decay parameter (𝜂𝑡) that can either ignore the last surprise (by setting 𝜂𝑡→0) or fully incorporate it (by setting 𝜂𝑡→1), depending on the context and relevance of the tokens. This allows Titans to adaptively manage memory based on the contextual significance of incoming tokens.

In contrast, SR-TTT's routing signal is closely related to reconstruction loss and its gradient magnitude, which is position-biased and requires a burn-in period. This suggests that SR-TTT may not be as flexible in adapting to changes in context as Titans, as it relies on a more static relationship between reconstruction loss and memory management.

Overall, Titans exhibit a more dynamic and context-sensitive approach to surprise and memory management compared to the more rigid structure observed in SR-TTT.

## Stage 3: prompt on ttt_rnn

**Question:** How does the TATS method improve video generation compared to traditional approaches?

### p0-original: faithfulness no claims made

The context does not provide enough information to explicitly explain how the TATS method improves video generation compared to traditional approaches.

### p2-step-by-step: faithfulness 1.00

The TATS method improves video generation by utilizing sliding window attention on the Transformer, allowing it to generate videos that exceed the length of the training data. This capability enhances the generation quality compared to traditional approaches, which may struggle with longer sequences (Source 2).

### p3-short-extractive: faithfulness 0.92

The TATS method improves video generation by proposing the sliding window attention on the Transformer to generate videos longer than the training length. This approach enhances generation quality significantly due to recent progress in auto-regression (AR) and diffusion-based approaches. The excerpts do not provide further details on how TATS compares to traditional approaches.

### p4-by-paper: faithfulness 0.94

**One-Minute Video Generation with Test-Time Training (2025)**: The TATS method improves video generation by proposing a sliding window attention mechanism on the Transformer, which allows for the generation of videos longer than the training length. This approach enhances generation quality significantly due to advancements in auto-regression and diffusion-based techniques (Source 2).

**tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction (2026)**: While this paper primarily focuses on 3D reconstruction, it highlights the importance of long-context autoregressive methods, which could imply a parallel in video generation techniques that benefit from similar long-context processing (Source 4).

The papers relate in that they both explore advanced techniques for handling long contexts in their respective domains—video generation and 3D reconstruction—suggesting a shared interest in improving performance through innovative memory and attention mechanisms.

### p1-strict-citations (chosen): faithfulness 1.00

The TATS method improves video generation by proposing the sliding window attention on the Transformer, which allows for the generation of videos longer than the training length (Source 2).

## Stage 4: model on hrm

**Question:** How does the performance of the ItrSA++ model with C-voting compare to the FPRM model in terms of solving Sudoku-Extreme, and what are the implications of their respective parameter counts?

### gpt-5-nano: faithfulness 1.00

Based on the excerpts provided, there is no information about a model named FPRM, nor any direct comparison between ItrSA++ with C-voting and FPRM on Sudoku-extreme. The papers discuss ItrSA++ with C-voting and compare its performance against AKOrN and HRM on Sudoku-hard, Sudoku-extreme, and Maze-hard (e.g., Figure 1(b) shows ItrSA++ with C-voting outperforming AKOrN on Sudoku-hard and Sudoku-extreme, and beating HRM on Sudoku-extreme in some contexts). They also discuss parameter counts for ItrSA++ (about 3 million parameters) and note this is about one-ninth the number of parameters of some baseline (implying larger baselines have around ~27 million), but there is no mention of FPRM’s performance or its parameter count in these excerpts.

What we can state from the excerpts:
- ItrSA++ with C-voting outperforms AKOrN on Sudoku-hard and Sudoku-extreme tasks in the presented comparisons, and outperforms HRM on Sudoku-extreme (and Maze-hard) in some figures (Source 4, 1(b); Source 2, 1).
- ItrSA++ with C-voting achieves high board accuracy with a model that has about 3 million parameters, which is about one-ninth the number of parameters of the compared baseline model referenced in the text (Source 4).
- The excerpts do not provide any data on FPRM’s performance on Sudoku-extreme, nor its parameter count, so a direct comparative performance assessment or implications about parameter counts between ItrSA++ with C-voting and FPRM cannot be stated from these sources.

If you have access to the FPRM results section in the same set of excerpts or in another part of the paper, I can synthesize a precise comparison. Otherwise, with the given excerpts, the requested comparison cannot be answered. (Source 1, Source 2, Source 3, Source 4, Source 5, Source 6, Source 7, Source 8)

### gpt-oss-20b: faithfulness no claims made

The excerpts you provided do not mention an FPRM model or any comparison between ItrSA++ with C‑voting and FPRM on Sudoku‑Extreme. Therefore I cannot answer how their performances compare or discuss implications of their parameter counts. If you have additional excerpts that include FPRM, please share them and I’ll be happy to help.

### gemini-3.5-flash-lite: faithfulness no claims made

Based on the provided excerpts, there is no mention of the FPRM model or its performance and parameter count. The excerpts only compare ItrSA++ to HRM and AKOrN. Therefore, the provided text does not contain enough information to answer your question.

### gpt-4o-mini (chosen): faithfulness no claims made

The excerpts do not provide any information regarding the performance of the ItrSA++ model with C-voting compared to the FPRM model in solving Sudoku-extreme. Additionally, there is no mention of the parameter counts for the FPRM model. Therefore, I cannot answer your question based on the provided excerpts.

## Stage 4: temperature on hrm

**Question:** How do the state update mechanisms differ between the Spherical Fixed-Point Reasoners model and the Hierarchical Reasoning Model?

### t-0.1: faithfulness 1.00

The excerpts do not provide enough information to directly compare the state update mechanisms between the Spherical Fixed-Point Reasoners (SFPRM) model and the Hierarchical Reasoning Model (HRM). While SFPRM utilizes a damped update through spherical linear interpolation (Slerp) and operates on the unit hypersphere, the excerpts do not detail the specific state update mechanisms employed by HRM. Therefore, a clear comparison cannot be made based on the provided information.

### t-0.4: faithfulness 0.67

The excerpts do not provide sufficient information to directly compare the state update mechanisms between the Spherical Fixed-Point Reasoners (SFPRM) model and the Hierarchical Reasoning Model (HRM). The SFPRM utilizes a damped update that is a spherical linear interpolation (Slerp) based on angular movements on the unit hypersphere (Source 5, Source 7). However, the excerpts do not detail the specific mechanisms of state updates for the HRM, only mentioning its failure modes and reasoning patterns (Source 6). Therefore, a clear comparison cannot be made based on the provided excerpts.

### t-0.7: faithfulness 0.75

The state update mechanisms of the Spherical Fixed-Point Reasoners (SFPRM) model and the Hierarchical Reasoning Model (HRM) differ in their approach to handling state updates and the conditions under which they operate.

In SFPRM, the state update is performed through a damped update process that utilizes spherical linear interpolation (Slerp). This involves moving a fraction of the way towards the proposal along a great circle on the unit hypersphere. The updates are guided by angles, and the halting criteria are based on the angular distance between the current state and the proposal (Source 5, Source 7). This method is characterized by a geometric approach that ensures boundedness and stability, as every hidden state is confined to the unit hypersphere.

In contrast, the HRM exhibits a failure mode when dealing with extremely simple puzzles, which is attributed to a violation of the fixed point property, a fundamental assumption of the model. This indicates that the HRM's state updates may not adapt effectively to certain conditions, which can lead to its inability to improve answers in a uniform manner (Source 6). The reasoning pattern in HRM does not involve a geometric framework but rather relies on a different mechanism that can result in inconsistencies in reasoning under certain circumstances.

Overall, SFPRM's updates are defined by geometric operations on the hypersphere, whereas HRM struggles with fixed point violations that affect its reasoning capability.

### t-1.0: faithfulness 1.00

The excerpts do not provide enough information to directly compare the state update mechanisms between the Spherical Fixed-Point Reasoners model (SFPRM) and the Hierarchical Reasoning Model (HRM). While SFPRM is described in terms of operations on the unit hypersphere, specifically using damped updates and spherical linear interpolation (Slerp) to manage state updates (Source 5, Source 7), there is no detailed description of the state update mechanism in HRM, making it impossible to outline differences based on the given excerpts.

### t-0.0 (chosen): faithfulness 1.00

The excerpts do not provide enough information to directly compare the state update mechanisms between the Spherical Fixed-Point Reasoners (SFPRM) model and the Hierarchical Reasoning Model (HRM). While SFPRM utilizes a damped update through spherical linear interpolation (Slerp) and operates on the unit hypersphere, the specific mechanisms of state updates in HRM are not detailed in the provided excerpts. Therefore, a clear comparison cannot be made based on the available information.
