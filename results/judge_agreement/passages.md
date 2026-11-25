# Claims to label

For each claim, decide from the passages alone whether it is SUPPORTED, PARTIAL or NOT_SUPPORTED, and write that in the human_verdict column of to_label.csv.

## hrm/q09
**Question:** What is the significance of using an ensemble of checkpoints from the training phase in improving model accuracy, and what unexpected result was observed from this approach?

**Claims:**
- c01: Picking 10 checkpoints from the later half of the training phase separated by approximately 1000 steps serves as an ensemble.
- c02: The steps between the checkpoints diversify their parameters.
- c03: The model bootstrapping method observed a 9.2% improvement in accuracy.
- c04: The final strongest checkpoint should cover the capability of all its predecessors.

**Passages the judge saw:**

[Source 1: Are Your Reasoning Models Reasoning or Guessing? A Mechanistic Analysis of Hierarchical Reasoning Models (2026)]
steps between them diversifies their parameters. Inspired
by this, we pick 10 of the checkpoints from the later half of
training phase separated by approximately 1000 steps as an
ensemble.
As we are testing the ensemble of checkpoints in the same
training run, which should be strongly correlated, one might
expect that the final strongest checkpoint (receiving the most
training) should cover the capability of all its predecessors.
However, we do observe a 9.2% improvement in accuracy
with this model bootstrapping method. This strongly contra-
dicts intuition, demonstrating that these mutually dependent
models actually differ significantly, in terms of which of the
test samples they get right.
4.6. Reasoning or Guessing?
The discoveries in Section 4.2 have already showed that

---

[Source 2: Evaluating Tiny Recursive Models Across Training for Code Generation (2026)]
therefore harms what the model predicts under teacher forcing
far more than what it generates: fit and generation decouple
within a single model and a single training run.
e) Single-snapshot artifacts (RQ1): This within-model
decoupling, together with the iso-depth double crossover
above, shows directly how single-checkpoint readings of these
runs mislead. A mid-training snapshot (epochs 8–28) would
report “the larger model fits better.” The epoch-40 snapshot
would report “the recursive model fits better.” Neither single
reading is a faithful summary, and the second is an over-
fitting artifact: TRM-AR regains the fit lead at epoch 40
only because vanilla-20L has degraded past its optimum, not
because recursion has improved relative to it. Validation-based

---

[Source 3: Evaluating Tiny Recursive Models Across Training for Code Generation (2026)]
optima are right-censored (Section VI). Throughout, we re-
port the direction and across-seed sign-consistency of each
effect rather than significance tests (Section VI). Results are
organized by comparison: the iso-parameter contrast addresses
parameter efficiency and generation quality (RQ2, RQ4); the
iso-depth contrast, the generalization analysis, and the closing
single-snapshot analysis address the across-training question
(RQ1); and the per-step compute cost (RQ3) is measured in
Section III-F. Section V answers each question explicitly. The
trajectory itself is the first result: the fit ranking between TRM-
AR and vanilla-20L reverses twice within the budget (epochs 8
and 40), so which arm “fits better” depends on the epoch at

---

[Source 4: Evaluating Tiny Recursive Models Across Training for Code Generation (2026)]
choice. The results also carry a methodological point. A single-
checkpoint fit comparison between TRM-AR and vanilla-20L
names a different winner at different epochs; reading each arm
at its validation-selected optimum, against the full trajectory,
yields one stable ranking. This pattern echoes the rationale for
released checkpoint suites in training-dynamics research [19]:
the point during training at which a comparison is made can
change its conclusion, so recursive models should be evaluated
on fit and generation jointly, across training.
VI. LIMITATIONS
This study is a controlled comparison of three architectures
on a shared instruction-to-code corpus across training, not a
benchmark of absolute code-generation ability. Its limitations
follow the standard validity dimensions.

---

[Source 5: Dynamical Systems Theory Behind a Hierarchical Reasoning Model (2026)]
H
. This entire (NLNH)–timestep process represents a single forward pass of the HRM.
2.2
Tiny Recursion Model
Tiny Recursive Models (TRMs), proposed in [7], represent a simple recursive reasoning approach using a single tiny
network recursing on its latent reasoning feature and progressively improving its final answer. The L-module and
H-module are represented by the same neural network (i.e., θL ≡θH and ˆFL ≡ˆFH). This single module consists of
two transformer layers (see Figure 2) with a self-attention layer or an MLP-mixer (MLP) [11], which is applied on
the sequence length. It significantly reduces the number of parameters by halving the number of layers and replacing
the two networks with a single tiny network. Another feature of the TRM that mitigates overfitting and improves the

---

[Source 6: Evaluating Tiny Recursive Models Across Training for Code Generation (2026)]
because recursion has improved relative to it. Validation-based
checkpoint selection together with the full trajectory resolves
the contradictory snapshots into a consistent result, in which
each arm is read at its validation-selected optimum and the
late crossing is recognized as degradation rather than a win.
V. DISCUSSION
Taken together, the results give a two-sided answer: which
model wins is time-indexed, and validation selection across
the full trajectory resolves it into a stable comparison. We first
summarize how the results answer each research question, then
turn to interpretation and implications.
2
8
16 20
40
Epoch (log scale)
1.2
1.5
3
10
Validation loss (log)
ep8
ep40
(a) Teacher-forced fit
2
8
16 20
40
Epoch (log scale)
0.0
0.5
1.0
1.5
2.0
2.5
3.0
Judge overall
ep8

---

[Source 7: Dynamical Systems Theory Behind a Hierarchical Reasoning Model (2026)]
To mitigate overfitting and improve generalization, the training of TRM employs an exponential moving average of the
weights. Another approach to prevent overfitting and simultaneously accelerate training and inference is the aggressive
reduction of the total number of model parameters. Furthermore, for the CMM, which operates as a contractive mapping
system, bounded nonlinear activation functions such as tanh may be preferable to unbounded functions like SiLU to
ensure stability and convergence.
Numerical Experiments
The performance of the proposed methods was evaluated on the Sudoku-Extreme dataset [1] as the main dataset. The
neural networks and their training were implemented in the PyTorch framework [29] (version 2.6 under CUDA 12.4).

---

[Source 8: Evaluating Tiny Recursive Models Across Training for Code Generation (2026)]
per optimizer step (Section III-F). The larger depth-matched
control is expected to fit better at its validation optimum, since
parameter count, not depth alone, governs how much a model
can store [28]. Our contribution is to reshape that ranking
across training and under generation.
C. Metric and Checkpoint Sensitivity in Evaluation
Teacher-forced next-token prediction can diverge from free-
running generation, so low held-out loss does not by itself
imply good generated output [17]. This train–inference mis-
match was recognized early in neural sequence modelling
and motivated corrections such as scheduled sampling and
sequence-level training [29], [30]. For code in particular, the
standard target is execution-based functional correctness such

## hrm/q15
**Question:** How does the dual phase–frequency state design in Winfree Oscillatory Neural Networks contribute to its performance in reasoning tasks?

**Claims:**
- c05: The dual-state design separates fast oscillatory dynamics from slower feature evolution.
- c06: The phase state undergoes multiple recurrent updates for rapid synchronization and local reorganization.
- c07: The frequency state is updated once per layer to act as a slower carrier of information.
- c08: WONN achieves competitive Maze-hard performance using only 1% of the parameters of the state-of-the-art HRM.

**Passages the judge saw:**

[Source 1: Winfree Oscillatory Neural Network (2026)]
parameterizations, hierarchical grouped synchronization dynamics, and a dual phase–frequency state
design, WONN provides a scalable framework for oscillatory neural computation.
Empirically, we show that synchronization-based architectures can scale far beyond the limited
settings previously explored in oscillatory or synchronization-inspired models. On visual perception
tasks, WONN achieves competitive or superior performance on CIFAR and ImageNet benchmarks
while maintaining strong parameter efficiency. In particular, WONN successfully scales to ImageNet-
1K, which, to the best of our knowledge, makes it the first synchronization-based oscillatory ar-
chitecture to achieve competitive large-scale image recognition performance. The learned phase

---

[Source 2: Winfree Oscillatory Neural Network (2026)]
phase organization, where distinct synchronized phase groups specialize to complementary coarse
and fine image structures. On complex reasoning tasks, WONN achieves competitive Maze-hard
performance using only 1% of the parameters of the state-of-the-art HRM [44], while also reach-
ing perfect test accuracy on Sudoku with strong parameter efficiency. Taken together, these results
suggest that generalized Winfree synchronization dynamics provide a scalable, high-performance,
and parameter-efficient computational principle for both visual perception and logical reasoning.
Winfree Oscillatory Neural Networks
This section develops WONN by connecting classical Winfree dynamics with learnable neural

---

[Source 3: Winfree Oscillatory Neural Network (2026)]
use it as a proxy for solution quality and perform voting to improve performance.
Separation of timescales. WONN separates fast oscillatory dynamics from slower feature evolution
through its dual-state design. Within each layer, the phase state Θ undergoes multiple recurrent
updates, enabling rapid synchronization and local reorganization on (S1)d. In contrast, the frequency
state Ωis updated once per layer, acting as a slower carrier of information across layers. This separa-
tion of timescales decouples transient oscillatory computation from cross-layer feature transformation,
which is reminiscent of hierarchical processing in biological systems [13, 15, 9, 44].
Experiments
3.1
Experimental Setup

---

[Source 4: Winfree Oscillatory Neural Network (2026)]
(4)
The dynamics define a vector field on M, while updates can be locally represented in the tangent
space TΘM ∼= Rd. In practice, this suggests performing computations in a Euclidean embedding
while enforcing periodic structure in the phase variables.
2.2
Winfree Oscillatory Neural Network
Fig. 2 illustrates the overall architecture of WONN, where representations are computed through
iterative synchronization dynamics between coupled oscillatory neurons.
State and initialization. Each layer maintains a dual state consisting of a phase variable Θ ∈(S1)d
and a frequency variable Ω∈Rd. The phase captures oscillatory structure, while the frequency
carries input-dependent information. Given an input x, we initialize Ω(0) = Ωinit = finit(x) and

---

[Source 5: Winfree Oscillatory Neural Network (2026)]
Models
Accuracy (%)
# Parameters
SAT-Net
98.3
0.618M
RRN
99.8
0.201M
R-Transformer
100.0
0.211M
Transformer
98.6± 0.3
16.87M
HRM
99.7± 0.2
27.28M
AKOrNattn(T = 16)
99.8± 0.1
2.98M
S(θi)&I(θi) as trigonometric functions
WONN (T = 16)
100.0± 0.0
1.58M
Conclusion
We introduced the Winfree Oscillatory Neural Network (WONN), a neural architecture built upon
generalized Winfree synchronization dynamics. Unlike conventional architectures that primarily rely
on static feature transformations, WONN performs computation through the collective evolution and
synchronization of phase oscillators on a toroidal state space (S1)d. By combining flexible interaction
parameterizations, hierarchical grouped synchronization dynamics, and a dual phase–frequency state

---

[Source 6: Winfree Oscillatory Neural Network (2026)]
dynamics serve as a scalable and parameter-efficient alternative to conventional feed-forward or
transformer-style neural architectures?
In this work, we introduce the Winfree Oscillatory Neural Network (WONN), a neural architecture
built upon generalized Winfree synchronization dynamics [46, 25]. As illustrated in Fig. 1, WONN
evolves neural representations on a high-dimensional toroidal phase space (S1)d, where each neuron
is represented as a phase oscillator. Unlike Kuramoto-type models based primarily on pairwise
phase differences, Winfree dynamics decompose interactions into separable sensitivity and influence
functions, enabling more flexible synchronization behaviors and interaction structures.

---

[Source 7: Winfree Oscillatory Neural Network (2026)]
Winfree Oscillatory Neural Network
Jiawen Dai3 and Yue Song†1,2
1Shanghai Qi Zhi Institute
2College of AI, Tsinghua University
3Shanghai Jiao Tong University
Abstract
Oscillations and synchronization are widely believed to play a fundamental role in
representation and computation. However, existing machine learning approaches
based on synchronization dynamics have largely been confined to specialized set-
tings such as object discovery, with limited evidence of scalability to standard vision
benchmarks or logic reasoning tasks. We propose the Winfree Oscillatory Neural
Network (WONN), a dynamical neural architecture based on generalized Winfree
dynamics. WONN evolves representations on the torus (S1)d through structured

---

[Source 8: Winfree Oscillatory Neural Network (2026)]
This section develops WONN by connecting classical Winfree dynamics with learnable neural
architectures. We first review the underlying Winfree model and its generalizations (Sec. 2.1), then
introduce the WONN architecture with structured interactions (Sec. 2.2), and finally discuss its key
dynamical and representational properties (Sec. 2.3).
2.1
From Winfree Dynamics to Learnable Oscillatory Systems
Winfree Dynamics. We consider a population of coupled phase oscillators that are governed by the
Winfree model [46]:
˙θi = ωi + κ
N S(θi)
N
X
j=1
I(θj),
i = 1, . . . , N,
(1)
where θi ∈T := R/(2πZ) is the phase of the i-th oscillator, ωi ∈R is its natural frequency and
κ ∈R+ is the coupling strength. The interaction is mediated through a mean-field aggregation of the

## hrm/q26
**Question:** How does the design choice of using tied residual coefficients in the Fixed-Point Reasoning Model (FPRM) contribute to the stability of the recurrent carry path?

**Claims:**
- c09: The residual stream is free to grow indefinitely.
- c10: The Fixed-Point Reasoning Model (FPRM) uses tied residual coefficients ($\alpha_1, \beta_1$) and iteration-level coefficients ($\alpha_2, \beta_2$).
- c11: There is a boundedness theorem that couples these coefficients.
- c12: The recurrent carry path has a gain of $\alpha_2 \alpha_1^{2D}$.

**Passages the judge saw:**

[Source 1: Spherical Fixed-Point Reasoners: Adaptive Loop Halting on the Hypersphere (2026)]
and iteration-level coefficients (α2, β2):
zℓ
i = α1zℓ−1
i
+ β1f ℓ
θℓ
 Normpre(zℓ−1
i
)

,
ℓ= 1, . . . , 2D,
z0
i+1 = α2z2D
i
+ β2x.
(1)
Via a boundedness theorem that couples the four coefficients:,
β2 = 1 −α2α2D
1 ,
β1 = β2(1−α1)
1−α2D
,
0 ≤α1, α2 < 1,
(2)
ensures that the recurrent carry path (gain α2α2D
1 ) is balanced against the injected input. Given a
proposal ezi = Fθ(zi; x), FPRM damps linearly:
zi+1 = (1 −ηi)zi + ηiezi,
(3)
with a per-sample ηi ∈(0, 1] that is decayed on a patience rule whenever the residual stalls, and
halts on the relative residual:
rE,i = ∥zi−ezi∥∞
∥ezi∥∞+ϵ ,
∥u∥∞:= max1≤t≤T, 1≤j≤d |(ut)j|,
(4)
once rE,i < τE or the budget is exhausted. The coefficient coupling and the denominator of the

---

[Source 2: Looped Reasoning Finishes Earlier Than You Think (2026)]
interpretation explicit, viewing reasoning as the evolution of a latent state converging toward solution-
aligned equilibria (Blayney et al., 2026; Kim, 2026). Fixed-Point Reasoning Model (FPRM) uses
convergence of the recurrent state to a fixed point as an intrinsic stopping criterion, Attractor Models
define the prediction itself as the equilibrium of a learned refinement process, and Equilibrium
Reasoners (EqR) explicitly learns task-conditioned attractors whose stable fixed points correspond to
valid solutions, using multiple initializations to explore their attraction basins (Movahedi et al., 2026;
Fein-Ashley & Rashidinejad, 2026; Huang et al., 2026). In this view, reasoning is not just a sequence
of refinements but a recurrent trajectory through the learned dynamical landscape.

---

[Source 3: Spherical Fixed-Point Reasoners: Adaptive Loop Halting on the Hypersphere (2026)]
Looped transformers and fixed-point reasoning. A looped Transformer applies a weight-tied
network repeatedly to a state that is re-exposed to the same input x at every application: zi+1 =
Fθ(zi; x), for i = 0, 1, . . ., and predicts with a readout hϕ(zi). A convergent iteration reaches a
fixed point z⋆= Fθ(z⋆; x), the equilibrium view of Bai et al. (2019). The prediction is determined
by the fixed point, while the number of iterations needed to reach it is chosen per input. FPRM
(Movahedi et al., 2026) uses a pre-norm block of depth D (2D attention and MLP sub-layers). Be-
cause the residual stream is free to grow indefinitely, FPRM uses tied residual coefficients (α1, β1)
and iteration-level coefficients (α2, β2):
zℓ
i = α1zℓ−1
i
+ β1f ℓ
θℓ
 Normpre(zℓ−1
i
)

,
ℓ= 1, . . . , 2D,
z0

---

[Source 4: Spherical Fixed-Point Reasoners: Adaptive Loop Halting on the Hypersphere (2026)]
residual of the tied block itself, at the cost of coupled residual-rescaling coefficients that keep its
Euclidean residual stream bounded; iterating past the point where the answer has settled is itself
Table 1: Recent recurrent reasoners: HRM (Wang et al., 2025), TRM (Jolicoeur-Martineau, 2025),
URM (Gao et al., 2025), Attractor (Fein-Ashley & Rashidinejad, 2026), EqR (Huang et al., 2026)
and FPRM (Movahedi et al., 2026). “Appr.” means no numerical fixed-point solve is performed.
Method
Recurrence
Fix. pt.
State update and input
Halting / output
HRM
Fast L-loop + slow H-loop
Appr.
Euclidean (zL, zH); input drives L
ACT head; read zH
TRM
Shared net refines z, answer y
No
Euclidean; input re-read each step
ACT head; read y
URM
UT; fixed inner + ACT outer loop
No

---

[Source 5: Spherical Fixed-Point Reasoners: Adaptive Loop Halting on the Hypersphere (2026)]
rescaling coefficients, and the halting residual needs a normalization factor to be scale-invariant.
We introduce Spherical Fixed-Point Reasoning Models (SFPRM), which run FPRM’s damped solver
on nGPT’s (Loshchilov et al., 2025) unit sphere (Figure 1). The normalized Transformer (nGPT)
obtains stable dynamics by changing the geometry rather than adding control: every hidden state
lives on the unit hypersphere Sd−1 and each sub-layer is a learned angular move followed by renor-
malization, so the divergence FPRM must guard against cannot occur. Every control quantity of
the loop becomes an angle: the input is re-injected by moving a fraction λ of the great-circle arc
toward it, the damped update is a spherical linear interpolation (Slerp) whose coefficient ηi is an

---

[Source 6: Spherical Fixed-Point Reasoners: Adaptive Loop Halting on the Hypersphere (2026)]
halting test compares each proposal with the current iterate, so it never needs the fixed point z⋆
t .
(i) Simplicity: because the state cannot leave the sphere, boundedness is structural, and FPRM’s
coupled residual-rescaling coefficients, its boundedness theorem, and its relative-norm resid-
ual disappear rather than being re-proved;
(ii) guarantees: a fixed point exists from continuity and input injection alone, with no contraction
or Lipschitz assumption; angular damping stabilizes any fixed point whose tangent Jacobian
has real parts below one, with an explicit step-size window; and sufficiently strong injection
makes it a unique global attractor;
(iii) performance: on Sudoku-Extreme and Maze-Hard, SFPRM halts up to 2.7× earlier than
FPRM at equal or higher accuracy; and

---

[Source 7: Spherical Fixed-Point Reasoners: Adaptive Loop Halting on the Hypersphere (2026)]
so halting is a learned head rather than a property of the dynamics. Recent methods such as HRM
(Wang et al., 2025), URM (Gao et al., 2025), TRM (Jolicoeur-Martineau, 2025), and PTRM (Sghaier
et al., 2026) inherit this design. Deep equilibrium models (Bai et al., 2019; 2020; Winston & Kolter,
2020; Bai et al., 2021) replace the unrolled iteration by the fixed point of a weight-tied layer and
differentiate implicitly through it. Attractor models (Fein-Ashley & Rashidinejad, 2026) and equi-
librium reasoners (Huang et al., 2026) make the loop’s attractor explicit and halt on a solver tol-
erance or a residual-based selection, and FPRM (Movahedi et al., 2026) halts on a scale-invariant
residual of the tied block itself, at the cost of coupled residual-rescaling coefficients that keep its

---

[Source 8: Spherical Fixed-Point Reasoners: Adaptive Loop Halting on the Hypersphere (2026)]
exact accuracy; mean halt and number of ex-
amples halting. Bottom: per-sequence residual
(thin), median (heavy).
With the following experiments we answer three
research questions:
(RQ1) Accuracy. How does SFPRM compare to
FPRM and other recurrent reasoner base-
lines? Does constraining the recurrence on
the sphere cost accuracy? (Section 4.1)
(RQ2) Convergence and halting. How effectively
does SFPRM utilize the learned recurrence
under the angular halting criterion and is
the fixed-point unique? (Section 4.2)
(RQ3) Geometry of the computation. What does
the fixed-point iteration write into the
state, and does the spherical parameteriza-
tion make it readable? (Section 4.3)
Setup.
We evaluate SFPRM on puzzle bench-
marks: Maze-Hard, Sudoku-Extreme (Wang et al.,

## nested_learning/q04
**Question:** How does the memory unit M_t utilize the sequence of token embeddings to predict the next token in an auto-regressive language model?

**Claims:**
- c13: A memory unit $M_t$ is defined as a sequence-to-sequence mapping.
- c14: The forward pass learns an online predictor $M_t: \mathbb{R}^d 	o \mathbb{R}$.
- c15: The memory unit encodes patterns contained in the prefix token embeddings $oldsymbol{X}_{\le t}$.
- c16: The memory unit retrieves relevant information by minimizing an energy function.

**Passages the judge saw:**

[Source 1: Beyond Test-Time Training: Learning to Reason via Hardware-Efficient Optimal Control (2026)]
many of them are derived from associative memory mechanisms (Hopfield, 1982, 1984; Hopfield & Tank, 1985; Schmidhuber,
1992; Ramsauer et al., 2020). In this view, patterns associated with the context s1, · · · , s𝑡are stored in memory, and s𝑡+1 is
synthesized by retrieving information from the contextual representation through the minimization of an energy function.
To be more specific, let 𝑿= [𝒙1, · · · , 𝒙𝑛]⊤∈R𝑛×𝑑denote a sequence of token embeddings, and let 𝑿≤𝑡= [𝒙1, · · · , 𝒙𝑡]⊤
denote the prefix up to time 𝑡. A memory unit is a sequence-to-sequence mapping, whose forward pass can be formulated
as learning an online predictor 𝑀𝑡: R𝑑→R that encodes the patterns contained in 𝑿≤𝑡and retrieves relevant information

---

[Source 2: End-to-End Test-Time Training for Long Context (2025)]
with all of its self-attention layers removed, leaving only the MLP layers. Our toy baseline – blithely
applying this architecture to language modeling – is effectively a bigram since it has no memory of
previous tokens. Our goal is to understand the effect of TTT in isolation, without the confounding
of other sequence modeling components.
One way to give our baseline architecture some memory is to train it on the context. Similar to
standard pre-training, we can predict ˆpt and compare it to xt at every t = 1,...,T as an exercise.
Specifically, denote the baseline architecture as f with weights W , then the standard next-token
prediction loss at time t can be written as:
ℓt(W ) = CE(f (xt−1;W ),xt).
(1)

---

[Source 3: Beyond Test-Time Training: Learning to Reason via Hardware-Efficient Optimal Control (2026)]
optimal control layer within the model architecture – integrating test-time memorization, world modeling, model-based
RL, and planning in a single architectural framework, enabling models to reason over future trajectories at inference time
while maintaining RL as a persistent objective throughout all training stages (Hatamizadeh et al., 2025; Dong et al., 2025).
Background: Memory-Based Architectures
Given a sequence of 𝑡tokens s1, · · · , s𝑡, auto-regressive LLMs learn to predict the next token according to the conditional
distribution 𝑝LLM(s𝑡+1|s1, · · · , s𝑡). While a variety of architectures have been proposed to model this conditional probability,
many of them are derived from associative memory mechanisms (Hopfield, 1982, 1984; Hopfield & Tank, 1985; Schmidhuber,

---

[Source 4: End-to-End Test-Time Training for Long Context (2025)]
massive amount of experience into their brains, which preserve the important information while
leaving out many details. For language models, training with next-token prediction also compresses
a massive amount of data into their weights. So what if we just continue training the language
model at test time via next-token prediction on the given context?
This form of Test-Time Training (TTT), similar to an old idea known as dynamic evaluation [72, 60],
still has a missing piece: At training time, we were optimizing the model for its loss out of the box,
not for its loss after TTT. To resolve this mismatch, we prepare the model’s initialization for TTT via
meta-learning [38, 79, 58] instead of standard pre-training. Specifically, each training sequence is

---

[Source 5: End-to-End Test-Time Training for Long Context (2025)]
• Prefill: conditioning on T + 1 given tokens x0,x1,...,xT , where x0 is the Beginning of Sequence
(<BOS>) token.
• Decode: predicting a distribution ˆpT +1 over all possible instantiations of the next token.
The test loss is then CE( ˆpT +1,xT +1), where CE is the cross entropy and xT +1 is generated by nature.
MLP
MLP
Embedding
Softmax
Embedding
MLP
MLP
Embedding
Softmax
Embedding
Forward
Legend
Backward
Updated
64
128
Token index
5.25
5.50
5.75
6.00
Loss (log perplexity)
TF. with attn.
TF. w/o attn.
TTT-naive 
TTT-E2E b = 1
TTT-E2E b = 16
Figure 2. Toy example. Left: Given x1 and x2 as context, we want to predict the unknown x3. Our toy baseline,
a Transformer without self-attention (using only the upward arrows), is effectively a bigram since it has no

---

[Source 6: Memory Caching: RNNs with Growing Memory (2026)]
Linear Memory. Let us start with a simple but extreme case, where the segment size is 1 and also the recurrent
memory is a vector-valued value-less memory module1. In this setup, each token xt ∈Rd is considered as a
segment and so the memory state only stores this token: i.e.,
M(t)
= M(t)
| {z }
bK
+ kt = bK + xtWK,
(18)
where M(t)
acts as a bias term for the projection of input xt to key kt. Next, applying memory caching on this
formulation results in:
yt =
t
X
i=1
γ(i)
t M(i)
1 (qt) =
t
X
i=1
exp
 u⊤
t ki

Pi
ℓ=1 exp
 u⊤
t kℓ

|
{z
}
γ(i)
t
(bK + xiWK)
|
{z
}
M(i)
qt .
(19)
Given the fact that qt is a free parameter and a function of input, one can re-parametrize the above formulation
by letting M(i)
= v′
i:
yt =
 
t
X
i=1
exp
 u⊤
t ki

Pi
ℓ=1 exp
 u⊤
t kℓ
 v′
i
!
⊗σ (xtWQ) ,
(20)

---

[Source 7: Fast Weight Programming and Linear Transformers: From ML to Neurobiology (2026)]
2018; Mishra et al., 2018; Miconi et al., 2018; 2019; Munkhdalai et al., 2019; Sandler et al., 2021; Kirsch and
Schmidhuber, 2021; Huisman et al., 2023), and they are not specific to the transformer architecture or the
language modality.
From the metalearning perspective, it is not surprising that LLMs are capable of in-context learning, given
that the task of auto-regressive next-token prediction—underlying language modeling—precisely follows the
form required by metalearning: prediction with error feedback (the delayed-feedback version above); and the
internet-scale text could provide data necessary for such meta-training (Irie and Lake, 2025).
Box 2: Slow Weight Configuration Implementing Gradient Descent

---

[Source 8: Tiny Autoregressive Recursive Models (2026)]
Given a fixed autoregressive decoder block template and a fixed next-token objective, how should
iterative computation be allocated to maximize generalization per unit compute? We answer this in a
controlled autoregressive family that fixes the token stream and next-token loss, enforces standard
causal masking and KV-cache semantics, uses an identical decoder block, and excludes routing and
token-stream modifications. Within this family, architectures correspond to different unrollings of the
same block template.
Findings.
Across compute-matched experiments on character-level algorithmic tasks (addition,
copying, reversing), untied depth and a flat two-stream recurrent baseline yield the strongest general-

## nested_learning/q11
**Question:** What is the role of the meta-optimization process in the Dynamic Nested Hierarchy model, and how does it address the limitations of long-context reasoning?

**Claims:**
- c17: The meta-optimization process in the Dynamic Nested Hierarchy (DNH) model governs the state of the hierarchy over time.
- c18: The meta-optimization process optimizes a meta-loss ($L_{meta}$) that measures adaptation efficacy.
- c19: The meta-loss incorporates distribution shifts, task-specific objectives, and adaptation penalties.
- c20: The model's levels, nesting structures, dependencies, and update frequencies can autonomously adapt.

**Passages the judge saw:**

[Source 1: Dynamic Nested Hierarchies: Self-Evolution in ML for Lifelong Intelligence (2025)]
To address these limitations, this work proposes Dynamic Nested Hierarchies (DNH), an extension of NL that endows
models with self-evolution capabilities. DNH represents architectures as time-varying directed acyclic graphs where
levels, dependencies, and frequencies adapt via meta-optimization, drawing from neuroplasticity to facilitate lifelong
learning without fixed constraints. Mathematically, DNH optimizes a meta-loss incorporating distribution shifts,
ensuring convergence in non-stationary regimes as proven through regret bounds and expressivity analyses.
The remainder of this paper is structured as follows. Section 2 formalizes DNH, detailing its limitations over static NL

---

[Source 2: Dynamic Nested Hierarchies: Self-Evolution in ML for Lifelong Intelligence (2025)]
due to rigid level definitions.
A PREPRINT - NOVEMBER 20, 2025
2.2
Formal Definition of Dynamic Nested Hierarchies
To overcome these limitations, we define a Dynamic Nested Hierarchy as a time-varying directed acyclic graph (DAG)
Gt = (Vt, Et), where vertices Vt = {M(ℓ)
t }Lt
ℓ=1 represent memory modules (associative memories as per Definition 1 in
[4]) at time t, and edges Et encode nesting dependencies. The number of levels Lt is dynamic, and each module M(ℓ)
t
has an adaptable update frequency f (ℓ)
t
∈R+.
The state of the hierarchy at time t is governed by a meta-optimization process:
Gt+1 = arg min
G Lmeta(G; Gt, xt, ∆t),
(3)
where Lmeta is a meta-loss measuring adaptation efficacy, xt is the current input, and ∆t quantifies distribution shift

---

[Source 3: Nested Learning: The Illusion of Deep Learning Architectures (2025)]
we saw that “gradient descent with momentum” algorithm itself can be viewed as a 2-level optimization process where the
momentum term itself is an associative memory that compress the past gradients into its parameters.
3.2
Nested Optimization Processes
In the previous section, we provided examples to demonstrate how one can decompose a machine learning model into a
set of nested or multi-level optimization procedures. Next, we first present a formalization of nested learning problems and
then define Neural Learning Module–an integrated computational system that learns from data.
In previous sections, we decomposed the model into a set of optimization process. However, it is still unclear if we can

---

[Source 4: Dynamic Nested Hierarchies: Self-Evolution in ML for Lifelong Intelligence (2025)]
their operational principles.
3.1
Overview of Self-Evolution Mechanisms
Self-evolution in DNH models is achieved through a closed-loop meta-optimization framework that monitors perfor-
mance metrics and triggers structural changes. At the core is a meta-controller that evaluates the hierarchy’s efficacy
using a composite loss function incorporating task-specific objectives and adaptation penalties. Formally, for a DNH
hierarchy Gt = (Vt, Et) at time t, the self-evolution process is defined as:
Gt+1 = Eϕ(Gt, xt, Lt),
(11)
where Eϕ is an evolution operator parameterized by ϕ, xt is the input data, and Lt is the instantaneous loss. The
parameters ϕ are learned via meta-gradient descent:
ϕt+1 = ϕt −ηϕ∇ϕE[Lmeta(Gt+1; xt+1)],
(12)

---

[Source 5: Dynamic Nested Hierarchies: Self-Evolution in ML for Lifelong Intelligence (2025)]
by dynamically compressing context flows and adapting to distribution shifts. Through rigorous
mathematical formulations, theoretical proofs of convergence, expressivity bounds, and sublinear
regret in varying regimes, alongside empirical demonstrations of superior performance in language
modeling, continual learning, and long-context reasoning, dynamic nested hierarchies establish a
foundational advancement toward adaptive, general-purpose intelligence.
Keywords Dynamic nested hierarchies · self-evolving models · lifelong learning · nested optimization · machine
learning architectures · neuroplasticity-inspired AI · continual adaptation · foundational models
Introduction
Advancements in deep learning have propelled machine learning models to unprecedented capabilities in domains

---

[Source 6: Nested Learning: The Illusion of Deep Learning Architectures (2025)]
when the initial value of memory is not meta-learned, which is common in earlier variants of linear transformers, the
gating of linear attention acts as a persistent memory and the initialization of the memory module.
5.1
Revisiting the Human Brain Perspective of Nested Learning
In Section 1.1, we discussed how structure in human brain is uniform and reusable, and if we need a new architecture in
deep learning, or if our beliefs about the heterogeneity of current models need to be revisited. In the previous sections,
we observed that both optimization process of neural networks as well as neural architectures can be formulated as a set
of nested and/or parallel optimization problems, in which the memory structure is a feedforward layer (e.g., either Deep

---

[Source 7: Dynamic Nested Hierarchies: Self-Evolution in ML for Lifelong Intelligence (2025)]
continual adaptation and lifelong learning. Building upon the nested learning paradigm, which
decomposes models into multi-level optimization problems with fixed update frequencies, this work
proposes dynamic nested hierarchies as the next evolutionary step in advancing artificial intelligence
and machine learning. Dynamic nested hierarchies empower models to autonomously adjust the
number of optimization levels, their nesting structures, and update frequencies during training or
inference, inspired by neuroplasticity to enable self-evolution without predefined constraints. This
innovation addresses the anterograde amnesia in existing models, facilitating true lifelong learning
by dynamically compressing context flows and adapting to distribution shifts. Through rigorous

---

[Source 8: Nested Learning: The Illusion of Deep Learning Architectures (2025)]
that generalizes the traditional viewpoint of “long-term/short-term memory blocks”. Our Hope architecture based on a
self-modifying Titans and CMS improves continual learning and long-context reasoning capabilitites, while remaining
competitive as a general backbone.
Is Catastrophic Forgetting Solved? While Hope and CMS have shown promising results in reducing catastrophic
forgetting in the tasks we empirically studied, the undesirable phenomenon of catastrophic forgetting is not “solved” in
general. From nested learning viewpoint on learning and backpropagation process, catastrophic forgetting is a natural
consequence of compression, where the limited capacity of the network forces the model to forget so that it retains capacity

## nested_learning/q36
**Question:** How do the concepts of fast weights in Fast Weight Programming (FWP) and test-time training (TTT) relate in terms of their approach to updating model parameters?

**Claims:**
- c21: Test-time training (TTT) can be viewed as a special case of fast weights.

**Passages the judge saw:**

[Source 1: End-to-End Test-Time Training for Long Context (2025)]
problem, with minimal changes to the architecture.
4.3
Fast Weights and Fast Weight Programmers
The general idea of fast weights is to update the parameters of a “fast” model on only the most
relevant data, as opposed to the conventional practice of updating a “slow” model on all data [94].
This idea has existed since the 1980s [26, 38, 97]. Because the most relevant data can often include
the test instance itself, test-time training can be viewed as a special case of fast weights, with a
heavier emphasis on the formulation of an explicit learning problem.
The general idea of fast weight programmers (FWPs) is to update the fast weights at test time with
a “slow” model (as a programmer) that, in turn, is updated less frequently, if at all [79]. In our

---

[Source 2: Fast Weight Programming and Linear Transformers: From ML to Neurobiology (2026)]
to weights of another (sub)network, called fast net, as a function of a sequence of input observations. By
conceptualizing weights of a neural network as its program/software (Schmidhuber, 1990a), such a slow net
which modifies the weights of a fast net is a programmer. The weights of the fast net are fast, because they
can rapidly change as a response to observations received at every time step, while those of the slow net are
slow because they are typically trained by some learning algorithm that updates slow weights on the sequence
level (and they typically become fixed after training).5
In most of the existing and practical FWP models, both fast and slow nets are one-layer networks, and
the weight update rule is some outer-product based one. However, the general concept of fast weight

---

[Source 3: Fast Weight Programming and Linear Transformers: From ML to Neurobiology (2026)]
Published in Transactions on Machine Learning Research (12/2025)
2021; Milakov and Gimelshein, 2018), which significantly reduces memory requirement (crucial for GPU
efficiency) compared to the naive algorithm that explicitly stores K⊤Q ∈RT ×T in Eq. 12; we refer to Dao
(2023) for further details.
Fast Weight Programmer Neural Networks
Here we present the concept of fast weight programming, and the resulting sequence models as well as their
key properties.
3.1
Basic Instantiation: FWP with a purely additive update rule
Before introducing the general definition of fast weight programmers in the next Sec. 3.2. Here we first
provide the most basic instantiation of FWPs: an “FWP with a purely additive outer product update rule”

---

[Source 4: Fast Weight Programming and Linear Transformers: From ML to Neurobiology (2026)]
the weight update rule is some outer-product based one. However, the general concept of fast weight
programming is more general and has no such a restriction (Irie et al., 2021); in principle, more complex (e.g.,
deeper) slow/fast nets or other update rules could be used. In fact, the core concept of FWPs is the idea of
training a network to train (e.g., generate weights of) another network—an idea which has been rebranded as
“hypernetworks” (Ha et al., 2017) in the modern deep learning literature6. One common challenge though is to
deal with the high-dimensionality of ANN weights, which are typically too large to be directly parameterized
as outputs of an ANN (except for tiny networks (Gomez and Schmidhuber, 2005)). The use of outer product

---

[Source 5: Fast Weight Programming and Linear Transformers: From ML to Neurobiology (2026)]
weights, while jointly training both sets of weights through the same gradient descent learning algorithm).
It was only in the early 1990s that end-to-end differentiable and learnable synaptic modulation dynamics
above (originally called “fast weight controllers”), featuring the “programming” part—the ‘P’ in FWP—was
proposed (Schmidhuber, 1991; 1992b), as an alternative to the conventional recurrence (Sec. 2.1) and was also
computationally motivated by the idea of reducing the ratio of trainable parameters to temporally changing
variables in sequence processing networks (Schmidhuber, 1993b). The FWP concept has seen a recent
revival (Schmidhuber, AI Blog, 2021; Irie and Schmidhuber, 2022) mainly due to its formal connection to the

---

[Source 6: End-to-End Test-Time Training for Long Context (2025)]
prediction loss at time t can be written as:
ℓt(W ) = CE(f (xt−1;W ),xt).
(1)
We update W at test time for every t = 1,...,T , in sequential order, with gradient descent:
Wt = Wt−1 −η ∇ℓt(Wt−1),
(2)
where η is a learning rate, and W0 is the initial weights at test time. In the end, we simply output
ˆpT +1 = f (xT ;WT ). We illustrate this form of TTT in the left panel of Figure 2: Our toy baseline only
uses the upward arrows, while TTT adds the backward and horizontal arrows.
2.2
Learning to (Learn at Test Time)
We now consider how W0 – the initial weights after (training-time) training but before test-time
training – is obtained, within the scope of the same toy example. By definition, our test-time training

---

[Source 7: Fast Weight Programming and Linear Transformers: From ML to Neurobiology (2026)]
corresponds to the derivative (w.r.t. the fast net weights Wt) of the squared loss ||vt −Wtϕ(kt)||2
2 between
the “target” vt and the output Wtϕ(kt) that the fast net would produce if ϕ(kt) were fed to its input (which
is consistent with the idea of binding ϕ(kt) to vt by storing the corresponding key/value pair in the memory
matrix Wt); see Box 3 for further details. More generally, the update rules used in typical FWP models
have a corresponding local objective function, as summarized in the last column of Table 1. For example,
using a state/weight decay in the update rule corresponds to introducing the L2 regularization on the fast
weight matrix ||W ||2
F in the local objective function (disregarding constant factors), where the regularization
strengths are the weight decay factors.

---

[Source 8: End-to-End Test-Time Training for Long Context (2025)]
a “slow” model (as a programmer) that, in turn, is updated less frequently, if at all [79]. In our
method, the inner-loop weights W can be viewed as “fast” and the outer-loop weights θ as “slow”.
Therefore, our method can be viewed as a special case of FWPs [58]. Next, we briefly review some of
the literature on FWPs in the order of relevance.
Clark et al. [17]. This work is the most relevant to ours in methodology. Given a Transformer
baseline with full attention, they add an MLP layer as fast weights, whose initialization is trained
as slow weights along with the rest of the model. Similar to ours, their method updates the fast
weights by taking a gradient step on the next-token prediction loss computed over each chunk

## titans/q10
**Question:** What is the purpose of Hierarchical Voting in the context of augmented inference, and how does it compare to Flattened Voting based on the results presented?

**Claims:**
- c22: The purpose of hierarchical voting is to combine augmented inference and structured voting to accurately select the correct answer when it is present.
- c23: Hierarchical voting effectively identifies the correct solution through a two-stage aggregation process.
- c24: The two-stage process involves intra-transformation voting and global voting.
- c25: Intra-transformation voting involves grouping predictions by transformation and selecting the top-3 most frequent predictions within each group.

**Passages the judge saw:**

[Source 1: The Surprising Effectiveness of Test-Time Training for Few-Shot Learning (2024)]
passes a flat voting approach. Our hierarchical method approaches
oracle-level performance, demonstrating its effectiveness in accu-
rately selecting the correct answer when present.
4.5. Impact of Augmented Inference
To analyze the impact of augmented inference and voting,
we run several ablations: (1) Vanilla, which generates two
predictions without transformations or advanced voting; (2)
Transformed Inference, applying a single transformation
(Rotate, Transpose, or Flip) to measure its isolated effect;
(3) Hierarchical Voting, our full pipeline combining aug-
mented inference and structured voting; (4) Flattened Vot-
ing, which selects the top-2 predictions from a single voting
round over all generated outputs; and (5) Oracle, an upper
bound that selects the correct answer if present.

---

[Source 2: The Surprising Effectiveness of Test-Time Training for Few-Shot Learning (2024)]
transform-1
Figure 4. Augmented inference and hierarchical voting. We
use leave-one-out tasks and invertible geometric transformations
to obtain multiple equivalent versions of the task for augmented
inference. Predictions from these versions are aggregated with a
hierarchical voting strategy: first, voting is performed within each
transformation, and then the top candidates from each transforma-
tion undergo global voting to yield the top two predictions.
obtain multiple responses and select the best according to
a ranker, called self-consistency (Wang et al., 2023). How-
ever, this is not viable in ARC (where the output grid is
directly predicted) as there is no way to directly enforce
diversity across samples while ensuring coherence within

---

[Source 3: The Surprising Effectiveness of Test-Time Training for Few-Shot Learning (2024)]
directly predicted) as there is no way to directly enforce
diversity across samples while ensuring coherence within
samples. As an alternative self-consistency approach, we
try an augmented inference strategy that combines greedy
decoding with multiple versions of the input. Specifically,
we generate multiple prediction candidates by using geomet-
ric transformations. We then employ a hierarchical voting
strategy to determine the final prediction from the set of
generated candidates. This approach involves two stages of
voting to progressively narrow down the best candidates: (1)
Intra-transformation voting: Group predictions by their
corresponding transformation t. Within each group, select
the top-3 most frequent predictions. (2) Global voting:

---

[Source 4: The Surprising Effectiveness of Test-Time Training for Few-Shot Learning (2024)]
round over all generated outputs; and (5) Oracle, an upper
bound that selects the correct answer if present.
As shown, individual transformations are modestly effective
on their own (with Transpose performing worst), but their
aggregation improves results markedly. Hierarchical voting
further outperforms a flattened voting approach and closely
approaches the oracle’s accuracy, suggesting that our two-
stage aggregation effectively identifies the correct solution
when it is present.
4.6. Comparison to Other Systems
Following our experiments on 80 tasks, we present com-
prehensive results on the full ARC public evaluation set,
comparing our system against existing approaches. Our
analysis focuses on three key aspects: the impact of our
TTT methodology, the benefits of combining our approach

---

[Source 5: The Surprising Effectiveness of Test-Time Training for Few-Shot Learning (2024)]
all model sizes, TTT leads to significant improvements in
performance. We also observe that for smaller model sizes,
TTT effectively closes the performance gap, with the 1B
and 3B models achieving similar accuracy after TTT.
The Surprising Effectiveness of Test-Time Training for Few-Shot Learning
Rotate
Transpose
Flip
Vanilla
Flattened
Hierarchical
Oracle
10
30
Tasks Solved
Individual
Aggregated
18
22
29
Figure 7. Accuracy of different transformations and voting
schemes. While individual transformations generally perform
at a modest level and are comparable to one another, aggregat-
ing across them through voting yields substantial improvements.
Notably, a hierarchical voting strategy with two voting stages sur-
passes a flat voting approach. Our hierarchical method approaches

---

[Source 6: The Surprising Effectiveness of Test-Time Training for Few-Shot Learning (2024)]
corresponding transformation t. Within each group, select
the top-3 most frequent predictions. (2) Global voting:
Take the selected transformation-specific candidates from
the previous step and select the top-2 most frequent predic-
tions across all transformations. The augmented inference
pipeline is summarized in Figure 4 and full details of the
pipeline are in Appendix E.
4.3. Impact of TTT Design
In this section, we compare the final implementation of our
method with different design choices for TTT. FT serves as
the baseline, using only the fine-tuned model with demon-
strations in-context. No Transformations omits the aug-
mentation step. Direct I/O Data replaces in-context tasks
with the direct input-output task formulation (Section 3.1).

---

[Source 7: A Survey on Latent Reasoning (2025)]
on the fly? What are the principles that govern this induced recurrence, and what new
capabilities does it unlock? Conversely, for hidden-state methods, as reasoning chains extend,
how can a model maintain a coherent "state of mind" over vast temporal sequences without
succumbing to the bottleneck of ever-expanding memory? Can this temporal evolution be
reframed as a form of continuous online optimization, conceptually unifying this horizontal
expansion with the iterative vertical refinement seen in activation-based methods?
While both approaches enhance reasoning capabilities, they differ in implementation re-
quirements and deployment flexibility, offering distinct pathways toward more powerful latent
reasoning. The following sections of this paper will describe these parts in detail.

---

[Source 8: The Surprising Effectiveness of Test-Time Training for Few-Shot Learning (2024)]
Inference
One of the most common techniques to scale
inference-time compute is to use temperature sampling to
The Surprising Effectiveness of Test-Time Training for Few-Shot Learning
x0
y0
Test Task
?
?
?
?
x1
x2
x3
?
x3
?
x3
y1
y2
x0
x1
y0
y1
x2
x1
y2
y1
?
x3
Hierarchical 
Majority Vote
(top-2)
x2
x0
y2
y0
Leave-one-out Queries
Rule Based Augmentations
Identity
Horizontal

## titans/q28
**Question:** What advantage does MSCANet provide in handling SAR change detection compared to existing methods?

**Claims:**
- c26: MSCANet is a recent transformer- and convolution-based approach.
- c27: MSCANet is surpassed by ChangeTitans in SAR change detection performance.
- c28: Quantitative metrics for MSCANet are provided in Table VI.

**Passages the judge saw:**

[Source 1: Towards Remote Sensing Change Detection with Neural Memory (2026)]
Qualitative results. To complement the quantitative anal-
ysis, Fig. 8 presents qualitative comparisons on the SAR-
CD dataset. Due to the inherent speckle noise and radiomet-
ric variations in SAR imagery, many existing methods tend
to produce scattered false alarms or blurred boundaries. In
contrast, our method yields cleaner change maps with more
precise delineation of object contours, even in cluttered or low-
contrast regions. These visual results further demonstrate the
robustness of the proposed framework in handling the unique
challenges of SAR-based change detection.
TABLE VII: Ablation studies on the LEVIR-CD dataset. Rows
highlighted in purple indicate the default configuration. The
highest value is highlighted in bold.
Experiment
Method
IoU
F1
P
R
w/o
81.88
90.04
90.83
89.26

---

[Source 2: Towards Remote Sensing Change Detection with Neural Memory (2026)]
mIoU, and H represents Hausdorff distance.
challenges due to strong speckle noise and cross-sensor varia-
tions. As reported in Table VI, ChangeTitans achieves the best
performance among all compared methods, reaching an IoU of
95.63% and an F1-score of 97.76%. These improvements are
consistent across both metrics and surpass recent transformer-
and convolution-based approaches, such as FCCDN [89] and
MSCANet [24]. Notably, the proposed method attains these
gains while maintaining competitive parameter and computa-
tional complexity, highlighting its robustness and efficiency in
handling challenging SAR change detection scenarios.
Qualitative results. To complement the quantitative anal-
ysis, Fig. 8 presents qualitative comparisons on the SAR-

---

[Source 3: Towards Remote Sensing Change Detection with Neural Memory (2026)]
(256 × 256 pixels, 0.5m/pixel) collected over Hong Kong
from 2007 to 2014. It encompasses a wide range of change
types, including urban construction, vegetation dynamics, and
coastal reclamation, making it particularly suitable for general-
purpose and heterogeneous change detection tasks. SAR-CD is
based on the validation partition of a dataset constructed using
Sentinel-1 SAR imagery over North-East Finland. It contains
10,000 512 × 512 bi-temporal SAR image pairs, which are
cropped into 256 × 256 patches and re-divided into 32,000
training, 4,000 validation, and 4,000 test samples. All samples
are labeled with pixel-wise change masks generated through
simulated changes, enabling robust evaluation under SAR-
specific challenges such as speckle noise.

---

[Source 4: Towards Remote Sensing Change Detection with Neural Memory (2026)]
pixel-wise accuracy and structural consistency in the predicted
change maps.
IV. EXPERIMENTS
A. Experimental Setup
Datasets: We evaluate the proposed method on four publicly
available remote sensing change detection datasets: LEVIR-
CD [37], WHU-CD [38], LEVIR-CD+ [39], SYSU-CD [40]
and SAR-CD [41]. LEVIR-CD contains 637 very high-
resolution (1024 × 1024 pixels, 0.5m/pixel) bitemporal image
pairs collected from Google Earth, covering 20 regions in
the United States between 2002 and 2018. It includes 31,333
annotated building change instances, making it a widely used
benchmark for fine-grained urban change detection. WHU-
CD, derived from the WHU Building dataset, consists of
two aerial images of Christchurch, New Zealand (captured in

---

[Source 5: Towards Remote Sensing Change Detection with Neural Memory (2026)]
ticularly in regions with irregular object shapes and heteroge-
neous backgrounds. These results complement the quantitative
findings in Table V, confirming that the proposed framework
is also effective in non-building change scenarios.
E. Experiments on SAR-CD
Quantitative Results. We further evaluate the proposed
framework on the SAR-CD dataset, which poses additional
TABLE VI: Comparison results on the SAR-CD [41] dataset.
All metrics are reported as percentages (%), with the highest
value highlighted in bold.
Method
Venue
Params
(M)
FLOPs
(G)
IoU
F1
FC-SiamDiff [31]
ICIP 2018
4.38
1.35
86.07 92.52
FC-SiamConc [31]
ICIP 2018
4.99
1.55
93.39 96.58
BIT [10]
TGRS 2021
3.55
4.35
93.48 96.63
MSCANet [24]
JSTARS 2022
16.59
14.68
94.43 97.14
FCS [89]
ISPRS 2022
3.03
9.59
94.77 97.31
DED [89]

---

[Source 6: Towards Remote Sensing Change Detection with Neural Memory (2026)]
well-suited for dense prediction tasks in remote sensing change
detection.
E. Two-stream Convolutional Block Attention
Existing remote sensing change detection approaches often
subtract features from two temporal images [31] or apply fixed
similarity metrics [22], [32]. These strategies can mislabel
weather, lighting, or seasonal variations as genuine changes,
leading to high false alarms. Additionally, minor misalign-
ments in registration may generate spurious differences around
boundaries or structures, further compromising accuracy.
To tackle these challenges, we propose a learnable TS-
CBAM fusion module that leverages channel and spatial atten-
tions across two temporal streams, thereby capturing deeper
spatiotemporal patterns. As depicted in Fig. 2(c), TS-CBAM

---

[Source 7: Towards Remote Sensing Change Detection with Neural Memory (2026)]
sonal or lighting differences. Extensive experiments on four
benchmark datasets (LEVIR-CD, WHU-CD, LEVIR-CD+,
and SYSU-CD) show that ChangeTitans achieves state-of-the-
art performance, surpassing existing methods in accuracy and
robustness. Ablation studies confirm the importance of neural
memory and adaptive feature fusion in achieving these gains.
Despite these advances, change detection in remote sensing
remains challenged by persistent issues such as pseudo-change
suppression and fine-grained boundary localization, particu-
larly in scenarios with significant temporal gaps or cross-
sensor imagery. Future work will improve the model’s adapt-
ability to diverse sensing environments and explore lightweight
variants suitable for real-time deployment scenarios.

---

[Source 8: Towards Remote Sensing Change Detection with Neural Memory (2026)]
4.35
93.48 96.63
MSCANet [24]
JSTARS 2022
16.59
14.68
94.43 97.14
FCS [89]
ISPRS 2022
3.03
9.59
94.77 97.31
DED [89]
ISPRS 2022
3.10
9.90
94.94 97.40
FCCDN [89]
ISPRS 2022
6.31
12.52
95.15 97.51
AFCF-3D [6]
TGRS 2023
17.54
31.72
93.72 96.76
ChangeMamba [18]
TGRS 2024
84.70
179.32
90.40 94.96
ChangeTitans(ours)
-
27.15
30.39
95.63 97.76
Fig. 8: Qualitative results on the SAR-CD [41] datasets.
Predicted outputs are color-coded as follows: white for true
positives (TP), black for true negatives (TN), green for false
positives (FP), and red for false negatives (FN). In the figure,
B represents Boundary F1 score, T represents Trimap-based
mIoU, and H represents Hausdorff distance.
challenges due to strong speckle noise and cross-sensor varia-

## titans/q33
**Question:** How does the VTitans framework for remote sensing change detection relate to the design tradeoffs in hybrid memory architectures discussed in the context of large language models?

**Claims:**
- c29: ChangeTitans draws inspiration from the recent success of Titans in language tasks.
- c30: ChangeTitans addresses challenges in capturing long-range dependencies while maintaining computational efficiency.
- c31: The provided excerpts do not contain specific details regarding the broader design tradeoffs in hybrid memory architectures.
- c32: The context of large language models is discussed beyond adapting the Titans architecture from language tasks to vision for remote sensing change detection.

**Passages the judge saw:**

[Source 1: Towards Remote Sensing Change Detection with Neural Memory (2026)]
adaptively aggregates local low-resolution features under a
convex constraint, enabling structure-aware reconstruction and
enhancing robustness in challenging remote sensing scenarios.
V. CONCLUSION
This paper presents ChangeTitans, a remote sensing change
detection framework that builds on the Titans architecture
with neural memory to address the challenges of capturing
long-range dependencies while maintaining computational ef-
ficiency. By integrating the VTitans backbone, hierarchical
feature adaptation, and the TS-CBAM fusion module, the
model effectively captures both local and global contextual
information while reducing spurious variations from sea-
sonal or lighting differences. Extensive experiments on four
benchmark datasets (LEVIR-CD, WHU-CD, LEVIR-CD+,

---

[Source 2: Towards Remote Sensing Change Detection with Neural Memory (2026)]
the recent success of Titans in language tasks, we present
ChangeTitans, the Titans-based framework for remote sensing
change detection. Specifically, we propose VTitans, the first
Titans-based vision backbone that integrates neural memory
with segmented local attention, thereby capturing long-range
dependencies while mitigating computational overhead. Next,
we present a hierarchical VTitans-Adapter to refine multi-
scale features across different network layers. Finally, we intro-
duce TS-CBAM, a two-stream fusion module leveraging cross-
temporal attention to suppress pseudo-changes and enhance
detection accuracy. Experimental evaluations on four benchmark
datasets (LEVIR-CD, WHU-CD, LEVIR-CD+, and SYSU-CD)
demonstrate that ChangeTitans achieves state-of-the-art results,

---

[Source 3: Towards Remote Sensing Change Detection with Neural Memory (2026)]
1
Towards Remote Sensing Change Detection with Neural Memory
Zhenyu Yang, Gensheng Pei, Yazhou Yao, Tianfei Zhou, Lizhong Ding, Fumin Shen
Abstract—Remote sensing change detection is essential for
environmental monitoring, urban planning, and related applica-
tions. However, current methods often struggle to capture long-
range dependencies while maintaining computational efficiency.
Although Transformers can effectively model global context,
their quadratic complexity poses scalability challenges, and
existing linear attention approaches frequently fail to capture
intricate spatiotemporal relationships. Drawing inspiration from
the recent success of Titans in language tasks, we present
ChangeTitans, the Titans-based framework for remote sensing

---

[Source 4: Towards Remote Sensing Change Detection with Neural Memory (2026)]
datasets and geographical regions.
C. Network Architecture
The proposed ChangeTitans architecture is designed for
efficient and accurate remote sensing change detection, lever-
aging neural memory and segmented attention mechanisms.
As shown in Fig. 2, the model consists of four key compo-
nents: (a) VTitans backbone, a ViT-like [30] vision backbone
based on the Titans architecture, integrating neural mem-
ory for long-range context modeling. (b) VTitans-Adapter,
which transforms non-hierarchical VTitans features into multi-
scale hierarchical representations, drawing inspiration from the
works of [21]. (c) TS-CBAM fusion module, a two-stream
attention mechanism for effectively fusing bi-temporal features
while suppressing pseudo-changes, and (d) Decoder, a U-

---

[Source 5: Towards Remote Sensing Change Detection with Neural Memory (2026)]
sonal or lighting differences. Extensive experiments on four
benchmark datasets (LEVIR-CD, WHU-CD, LEVIR-CD+,
and SYSU-CD) show that ChangeTitans achieves state-of-the-
art performance, surpassing existing methods in accuracy and
robustness. Ablation studies confirm the importance of neural
memory and adaptive feature fusion in achieving these gains.
Despite these advances, change detection in remote sensing
remains challenged by persistent issues such as pseudo-change
suppression and fine-grained boundary localization, particu-
larly in scenarios with significant temporal gaps or cross-
sensor imagery. Future work will improve the model’s adapt-
ability to diverse sensing environments and explore lightweight
variants suitable for real-time deployment scenarios.

---

[Source 6: Towards Remote Sensing Change Detection with Neural Memory (2026)]
Fig. 6: Qualitative comparison of change detection results on
the LEVIR-CD [37] dataset under label noise. Here, label
noise refers to incomplete or incorrect annotations in the
training set, as highlighted in the
purple
boxed regions.
models such as STANet-PAM [37], ChangeStar [77], and
Changer [78], VTitans improves Top-1 accuracy by 5.8% with
an equivalent parameter budget. These results highlight the
effectiveness of the integrated neural memory mechanism. Un-
like CNN-based models, VTitans significantly enhances global
contextual modeling through memory-based representations.
In contrast to Transformer-based methods, VTitans combines
segmented attention for local feature extraction with memory-
based global reasoning, delivering strong performance while

---

[Source 7: Towards Remote Sensing Change Detection with Neural Memory (2026)]
each segment. Titans unifies these strengths, achieving fidelity
and efficiency in language modeling, a property we seek to
harness for vision.
Neural memory captures long-range context, while seg-
mented attention focuses on local structure. This allows effi-
cient reasoning over high-resolution scenes without quadratic
complexity. Guided by this principle, we present ChangeTi-
tans, the first Titans-style vision architecture for remote sens-
ing change detection. Its backbone, VTitans, follows a flat
Vision-Transformer layout: high-resolution images are divided
into segments, intra-segment attention refines fine textures, and
arXiv:2602.10491v1  [cs.CV]  11 Feb 2026
distilled segment tokens are written to an external memory so

---

[Source 8: Towards Remote Sensing Change Detection with Neural Memory (2026)]
4e-3, cosine annealing learning rate decay, and a batch size
of 64 per GPU across 4 NVIDIA RTX 3090 GPUs. A linear
warm-up phase is applied for the first 20 epochs to stabilize
convergence, after which the learning rate follows the cosine
schedule. Gradient clipping with a maximum norm of 1.0 is
applied to prevent exploding gradients due to the memory
update mechanism. No Exponential Moving Average (EMA)
is used during pre-training.
As reported in Table III, VTitans achieves a Top-1 accuracy
of 75.7% on ImageNet-1K, demonstrating competitive perfor-
mance relative to similar scale models. Compared to ResNet-
18, a commonly adopted in remote sensing change detection
Fig. 6: Qualitative comparison of change detection results on
the LEVIR-CD [37] dataset under label noise. Here, label

## ttt_rnn/q21
**Question:** What is the purpose of introducing fast weights in Test-Time Training, and how do they address the limitations of traditional recurrent neural networks?

**Claims:**
- c33: The purpose of introducing fast weights in Test-Time Training (TTT) is to dynamically store contextual information during inference/test-time through updates.
- c34: Updates can include self-supervised updates or mean-square error.
- c35: Traditional RNNs compress sequence context into a fixed-size hidden state.
- c36: The fixed-size hidden state of traditional RNNs is constrained in its representational capacity.

**Passages the judge saw:**

[Source 1: Learning What to Remember: Test-Time Training via Context Distillation (2026)]
2.1
Test-Time Training
Recurrent neural networks have a long history in AI [24, 8, 3]. Recent recurrent and test-time
training architectures use fixed-size states or fast-weight updates as sub-quadratic alternatives to full
attention [21, 16, 35, 33, 20, 12, 52, 50]. DeltaNet and Gated DeltaNet use delta-rule key-value state
updates [52, 50]; Mamba uses selective state-space recurrences [20]; and Log-Linear Attention and
RATTENTION interpolate between linear and softmax attention to improve in-context recall [22, 48].
While these models differ in their parameterization of the recurrent state, they share the view that
long-context computation can be compressed into a compact dynamical memory rather than explicitly
stored in the attention cache.

---

[Source 2: TTT3R: 3D Reconstruction as Test-Time Training (2025)]
between retaining historical memory and adapting to new information, as shown in Figure 3. The
iterate states are also known as fast weights [64, 65], as they change in-context with each timestep,
rapidly adapting to the input tokens. In contrast to the slow weights in neural networks—which act
as meta-learners [31] and are only adjusted during training—fast weights are learned to function as
associative memory [9, 60]. Recent examples of such layers include DeltaNet [64, 98], TTT [73, 103],
and Titans [6], each of these layers was derived from a specific choice of retention and adaptation.
The idea of learning at test time [10, 72] has also been explored in 3D reconstruction, where methods
such as CVD [48] and Test3R [100] fine-tune a pre-trained model on the test sequence to minimize

---

[Source 3: tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction (2026)]
the key-value pairs (ki, vi) of input tokens as training data
to update fast weights using mean-square error: W
←
W −η∇LMSE(fW (k), v), which can be further applied to
queries to obtain the final input o = fW (q). In this way, the
fast weights effectively encode the key–value (KV) cache
of the input sequence into a fixed-size neural memory.
Originally, the TTT model [47] updates the fast weights
using only a small minibatch (e.g. 16 tokens), which re-
sults in very low GPU FLOP utilization and difficulty in
handling long sequences. Large Chunk Test-Time Training
(LaCT) [70] instead updates fast weights with large chunk
size (up to 1M tokens). Its chunk-wise update computes the
gradient of the summed loss over all keys and values within
the chunk. More details can be found in [70].

---

[Source 4: Learning to (Learn at Test Time): RNNs with Expressive Hidden States (2024)]
model that is updated less frequently, if at all [65]. Our inner-loop weights W can be viewed as
“fast” and outer-loop weights θ as “slow”. Therefore, networks containing TTT layers can be viewed
as a special case of FWPs [46], similar to how TTT can be viewed as a special case of fast weights.
Notably, one instantiation in Irie et al. [38] makes the fast weights an MLP, preceding our TTT-MLP.
Many other modern RNN layers such as linear attention [44, 63] and DeltaNet [62, 83] are also
inspired by the idea of FWPs. Given their relevance to our work, we discuss these modern RNN
layers in detail in the next subsection. For the rest of this subsection, we briefly outline a few
other forms of FWPs. Clark et al. [15] give a Transformer a final layer of fast weights, whose

---

[Source 5: Learning to (Learn at Test Time): RNNs with Expressive Hidden States (2024)]
as face detection [41], object detection [56], image super-resolution [68], and 3D reconstruction [53].
More recently, the same idea has also been applied to natural language processing, where it is called
dynamic evaluation [47, 48]. The basic approach is to directly finetune a language model on the test
sequence, which often comes in the form of a prompt.
Next, we discuss two relevant lines of work in detail: test-time training and fast weights.
4.1.1
Test-Time Training
The core idea of Test-Time Training (TTT) is that each test instance defines its own learning problem,
where this test instance alone is the target of generalization [72]. Concretely, for each test instance x,
the conventional practice is to predict f (x), using a predictor f that is optimized for all training

---

[Source 6: Scal3R: Scalable Test-Time Training for Large-Scale 3D Reconstruction (2026)]
to capture long-range dependencies through the GCM mod-
ules, while preserving VGGT’s strong geometric reasoning
and scalability, making large-scale training across diverse
datasets possible.
4.2. Test-Time Training as Memory
Although the adaptable neural sub-networks substantially
enlarge memory capacity compared to the fixed-size state
token of traditional RNNs (as discussed in Section 3), exist-
ing TTT [5, 30, 67, 79] approaches still struggle to scale to
long contexts. This limitation primarily stems from ineffi-
cient small-batch updates and sub-optimal GPU utilization,
where frequent fine-grained updates hinder throughput and
constrain the maximum sequence length.
Inspired by recent work LaCT [100], which adopts ex-
tremely large chunks as the update unit in TTT to improve

---

[Source 7: Learning to (Learn at Test Time): RNNs with Expressive Hidden States (2024)]
The general idea of fast weights is to update the parameters of a “fast” model on only the most
relevant data, as opposed to the conventional practice of updating a “slow” model on all data [74].
This idea has existed since the 1980s [20, 32, 78]. The most relevant data often includes the test
instance itself, therefore TTT can be viewed as a special case of fast weights. Compared to fast
weights, TTT embraces the idea of formulating an explicit learning problem, where the test instance
is the target of generalization. Our update rule is also an explicit step of optimization.
The idea of fast weight programmers (FWPs) is to update the fast weights at test time with a “slow”
model that is updated less frequently, if at all [65]. Our inner-loop weights W can be viewed as

---

[Source 8: Scal3R: Scalable Test-Time Training for Large-Scale 3D Reconstruction (2026)]
vector. To alleviate the quadratic complexity of softmax
attention, recurrent neural networks (RNNs) and their vari-
ants [23, 27, 49, 49] compress sequence context into a fixed-
size hidden state ht ∈Rd. At each timestamp t, the hidden
state is updated based on the current input xt and the previ-
ous hidden state ht−1 as:
  h _t = \sig m a (\theta _{ss}h_{t-1} + \theta _{sx}x_{t}). 
(2)
However, this compression is inherently constrained by the
representational capacity of the hidden state, leading to in-
formation degradation, especially in long sequences.
To overcome this limitation,
Test-Time Training
(TTT) [67] introduces fast weights, a set of rapidly adapt-
able neural parameters W that dynamically store contextual
information during inference through self-supervised up-

## ttt_rnn/q35
**Question:** How do the probabilistic knowledge acquisition methods in language models relate to the iterative reasoning process proposed in Closed-Loop Transformers?

**Claims:**
- c37: The texts do not contain enough information to connect probabilistic knowledge acquisition methods directly to the iterative reasoning process proposed in Closed-Loop Transformers.
- c38: Probabilistic language modeling is the core loss during unsupervised pre-training for knowledge capability across various architectures.
- c39: Equilibrium Transformers and closed-loop prediction are separate mechanisms to overcome the commitment bottleneck of open-loop autoregression.
- c40: Equilibrium Transformers and closed-loop prediction perform latent reasoning.

**Passages the judge saw:**

[Source 1: Closed-Loop Transformers: Autoregressive Modeling as Iterative Latent Equilibrium (2025)]
reasoning from language models.
Consider the contrast with human cognition. When solving a difficult problem, humans do not commit to each word
irreversibly; they pause, reconsider, revise mental representations, and verify consistency before proceeding [19].
Chain-of-thought prompting [52] attempts to externalize this process in token space, but it remains a workaround rather
than an architectural solution, since the model still makes one-shot predictions for each token in the chain.
Equilibrium Transformers internalize deliberation at the representational level. The iterative refinement process (Eq. 8)
implements a form of latent reasoning: the model searches for self-consistent beliefs before committing to observable

---

[Source 2: Open Problems and a Hypothetical Path Forward in LLM Knowledge Paradigms (2025)]
including real-world knowledge. Sequence model architectures, not limited to transformers, but also
including RWKV (Peng et al., 2023) and Mamba (Gu and Dao, 2024), have demonstrated similar
levels of knowledge capability after pre-training. This suggests that knowledge acquisition is more
closely tied to the training paradigm than to the speciﬁc architecture. In this section, we primarily
discuss how language models represent knowledge from the perspective of architecture-agnostic
training paradigm, providing background and preliminaries for subsequent discussions.
Training Paradigm
The foundation of LLM knowledge lies in probabilistic language modeling,
which constitutes the core loss during the unsupervised pre-training phase of the model (Radford and Narasimhan,

---

[Source 3: Closed-Loop Transformers: Autoregressive Modeling as Iterative Latent Equilibrium (2025)]
implements a form of latent reasoning: the model searches for self-consistent beliefs before committing to observable
outputs. This is not merely a matter of increasing computation time; it corresponds to a qualitatively different process in
which inference proceeds in a closed-loop rather than open-loop fashion.
We hypothesize that this architectural shift will prove necessary, not merely beneficial, for achieving robust reasoning in
language models. Just as attention mechanisms [1] were necessary to overcome the sequential bottleneck of RNNs,
closed-loop refinement may be necessary to overcome the commitment bottleneck of open-loop autoregression.
7.2
Connections to Neuroscience and Cognitive Architecture

---

[Source 4: Closed-Loop Transformers: Autoregressive Modeling as Iterative Latent Equilibrium (2025)]
with long-range dependencies. The reverse prediction loss Lrev forces the model to maintain representations from which
the entire context can be reconstructed, preventing the forgetting that affects standard transformers.
Conclusion
This paper introduced the closed-loop prediction principle, a rethinking of autoregressive sequence modeling. We
argued that the dominant open-loop paradigm, in which each hidden state is computed in a single forward pass and
never revised, constitutes a critical architectural bottleneck that prevents transformers from achieving robust reasoning.
To address this limitation, we proposed Equilibrium Transformers, which replace one-shot inference with iterative
refinement toward self-consistent equilibria defined by a learned energy function.

---

[Source 5: Closed-Loop Transformers: Autoregressive Modeling as Iterative Latent Equilibrium (2025)]
systems that “think” via gradient-based belief revision.
While TTT provides the adaptation signal, iterative methods the reasoning loop, and DEMs the solver, none fully
close the autoregressive feedback loop end-to-end. Equilibrium Transformers synthesize these into a principled
architecture that guarantees convergence, scales asymptotically with Transformers, and unifies disparate paradigms
under closed-loop prediction.
CLOSED-LOOP TRANSFORMERS - JAFARI & ANBARJAFARI
Discussion and Implications
These preliminary results provide encouraging evidence for the closed-loop prediction principle. On a challenging
algorithmic task, Equilibrium Transformers demonstrate:
• Consistent improvements when the task is difficult (+3.28

---

[Source 6: Closed-Loop Transformers: Autoregressive Modeling as Iterative Latent Equilibrium (2025)]
CLOSED-LOOP TRANSFORMERS: AUTOREGRESSIVE MODELING
AS ITERATIVE LATENT EQUILIBRIUM
Akbar Anbar Jafari
University of Tartu
Tartu, Estonia
akbar.anbar.jafari@ut.ee
Gholamreza Anbarjafari
3S Holding OÜ
Tartu, Estonia
shb@3sholding.com
December 1, 2025
ABSTRACT
Contemporary autoregressive transformers operate in open loop: each hidden state is computed
in a single forward pass and never revised, causing errors to propagate uncorrected through the
sequence. We identify this open-loop bottleneck as a fundamental architectural limitation underlying
well-documented failures in long-range reasoning, factual consistency, and multi-step planning. To
address this limitation, we introduce the closed-loop prediction principle, which requires that models

---

[Source 7: Closed-Loop Transformers: Autoregressive Modeling as Iterative Latent Equilibrium (2025)]
• Noise schedules from diffusion models can inform the damping parameter γ in EqT.
• Amortization techniques from variational inference can accelerate the equilibrium solver.
• Consistency distillation [59] can reduce K while preserving quality.
• RLHF objectives can be incorporated as additional energy terms for alignment.
7.5
Potential Impact on Key Capabilities
We now discuss how the closed-loop principle may address specific failure modes of current language models.
7.5.1
Reasoning and Planning
Multi-step reasoning remains a significant challenge for transformers [14]. Errors in early reasoning steps compound
catastrophically, and models struggle to backtrack or revise incorrect intermediate conclusions. Equilibrium Trans-

---

[Source 8: Closed-Loop Transformers: Autoregressive Modeling as Iterative Latent Equilibrium (2025)]
before committing to outputs, is the next limitation that must be addressed. Equilibrium Transformers offer one concrete
instantiation of this principle, and the broader idea of closed-loop prediction is likely to inspire further architectural
developments.
CLOSED-LOOP TRANSFORMERS - JAFARI & ANBARJAFARI
Looking ahead, we envision a generation of language models that do not merely pattern-match their way through
sequences, but instead perform explicit reasoning by searching for self-consistent beliefs before generating outputs.
Such models would reduce hallucination by verifying factual coherence, improve multi-step reasoning by detecting and
correcting logical errors, and adapt their computational effort to problem difficulty. The path from current transformers
