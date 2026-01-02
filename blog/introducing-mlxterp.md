# Democratizing Mechanistic Interpretability: Bringing Neural Network Analysis to Apple Silicon

*How unified memory architecture and thoughtful API design are making interpretability research accessible to researchers everywhere*

**January 2025 | COAI Research Institute**

---

## Abstract

Mechanistic interpretability—the systematic study of how neural networks implement learned algorithms—has emerged as one of the most promising approaches to understanding and ensuring the safety of advanced AI systems. However, the computational requirements of interpretability research have historically limited participation to well-resourced institutions with access to expensive GPU clusters. In this article, we introduce mlxterp, an open-source library that leverages Apple Silicon's unified memory architecture to enable sophisticated interpretability research on consumer hardware. We discuss the scientific motivations behind mechanistic interpretability, analyze the unique advantages of unified memory for activation analysis, and demonstrate how mlxterp's API design facilitates seamless transitions between local experimentation and cloud-scale validation. Our goal is not merely to provide a tool, but to lower barriers to entry for a field that urgently needs broader participation.

---

## 1. Introduction: The Interpretability Imperative

The rapid advancement of large language models has created an unprecedented situation in the history of technology: we are deploying systems whose capabilities exceed our understanding of their internal mechanisms. A model like GPT-4 or Claude can engage in sophisticated reasoning, demonstrate apparent creativity, and exhibit behaviors that surprise even its creators—yet we lack fundamental insight into *how* these capabilities emerge from the model's learned parameters.

This epistemic gap poses significant challenges. Without understanding why a model produces particular outputs, we cannot reliably predict when it will fail, detect when it might be engaging in deceptive behavior, or confidently align its objectives with human values. The field of mechanistic interpretability has emerged as a response to these challenges, seeking to reverse-engineer neural networks much as biologists reverse-engineer biological systems or security researchers reverse-engineer software.

The core premise of mechanistic interpretability is that neural networks, despite their complexity, implement interpretable algorithms that can be discovered through careful analysis. This premise has been validated by numerous research successes: the discovery of "induction heads" that implement in-context learning (Olsson et al., 2022), the identification of circuits responsible for indirect object identification (Wang et al., 2022), and the extraction of features representing human-interpretable concepts from model activations (Cunningham et al., 2023).

Yet despite these successes, mechanistic interpretability remains a relatively small field, practiced primarily at a handful of well-funded organizations. The reasons are partly computational: serious interpretability work requires capturing and analyzing activations from models with billions of parameters, running numerous experiments with different interventions, and iterating rapidly through hypotheses. These requirements have traditionally demanded expensive cloud GPU access, creating barriers that exclude many potential contributors.

We believe this exclusion is not merely unfortunate but actively harmful to the field's progress. Interpretability research benefits enormously from diverse perspectives—researchers with different backgrounds notice different patterns, ask different questions, and develop different intuitions. By limiting participation to those with institutional resources, we are limiting the field's intellectual diversity at precisely the moment when it most needs fresh thinking.

mlxterp represents our attempt to address this problem by exploiting a technological development that makes high-quality interpretability research possible on consumer hardware: Apple Silicon's unified memory architecture.

---

## 2. Background: The Science of Mechanistic Interpretability

### 2.1 From Black Boxes to Mechanistic Understanding

Neural networks have long been described as "black boxes"—systems that transform inputs to outputs through processes too complex for humans to understand. This characterization, while historically reasonable, increasingly appears to be a limitation of our tools rather than an inherent property of the systems themselves.

The mechanistic interpretability research program is founded on the hypothesis that neural networks implement relatively simple algorithms, but do so in a distributed manner across many parameters. The goal is to identify these algorithms—termed "circuits"—and understand how they compose to produce the network's overall behavior.

Consider a simple example: how does a language model complete the phrase "The Eiffel Tower is located in"? A mechanistic account might reveal that:

1. Early layers identify "Eiffel Tower" as a named entity requiring geographic information
2. Middle layers retrieve the association between "Eiffel Tower" and "Paris" from the model's parametric memory
3. Late layers format this retrieved information as an appropriate completion

Each of these steps might be implemented by specific "circuits"—patterns of connectivity and computation that perform identifiable functions. Discovering these circuits allows us to understand not just *what* the model does, but *how* and *why* it does it.

### 2.2 Key Techniques in Mechanistic Interpretability

The field has developed several core techniques for investigating neural network internals:

**Activation Analysis** involves capturing the intermediate computations (activations) produced by a model during inference and analyzing their properties. This includes examining activation magnitudes, correlations between activations and input features, and how activations evolve through the network's layers. The "logit lens" technique (nostalgebraist, 2020) exemplifies this approach, projecting intermediate activations to vocabulary space to reveal what the model "believes" at each layer.

**Causal Interventions** go beyond passive observation to actively modify activations and observe the effects on model behavior. If zeroing out a particular component causes the model to lose a specific capability, this provides causal evidence that the component is involved in implementing that capability. Activation patching (Vig et al., 2020) and path patching (Goldowsky-Dill et al., 2023) are sophisticated variants that allow researchers to precisely localize the sources of model behaviors.

**Feature Extraction** attempts to identify interpretable directions in activation space. Sparse autoencoders (Cunningham et al., 2023; Bricken et al., 2023) have proven particularly effective, learning to decompose activations into sparse combinations of human-interpretable features. This technique has revealed that models learn rich representations of concepts like "code," "deception," and "uncertainty" that can be analyzed and potentially controlled.

**Circuit Discovery** synthesizes these techniques to identify complete computational subgraphs responsible for specific behaviors. The goal is to produce explanations at the level of "attention head A in layer 3 moves information from the subject token to the final position, where MLP neuron B looks up the associated attribute."

### 2.3 The Computational Challenge

These techniques, while conceptually straightforward, impose significant computational demands. Consider what's required to perform a simple activation patching experiment:

1. Run the model on a "clean" input and cache all intermediate activations
2. Run the model on a "corrupted" input, intervening to replace specific activations with their clean counterparts
3. Measure the effect on output probabilities
4. Repeat for each component of interest (potentially thousands of attention heads and MLP neurons)

For a model with 32 layers, each containing multiple attention heads and an MLP, a comprehensive patching sweep requires thousands of forward passes, each producing gigabytes of activation data. Multiply this by the number of examples needed for statistical reliability, and the computational requirements become substantial.

Traditional GPU computing imposes additional overhead: activations must be copied from GPU memory to CPU memory for analysis, creating a bottleneck that slows iterative research workflows. Researchers frequently report spending more time managing memory and waiting for data transfers than actually analyzing results.

---

## 3. The Apple Silicon Opportunity

### 3.1 Unified Memory Architecture

Apple Silicon's most significant innovation for machine learning workloads is not its raw computational speed but its memory architecture. Traditional computing systems maintain separate memory pools for CPU and GPU, requiring explicit data transfers between them. This separation made sense historically—GPUs were discrete cards with their own DRAM, physically separate from the CPU's memory.

Apple's system-on-chip design eliminates this separation. The M1, M2, M3, and M4 families feature unified memory that is directly accessible to all processing units—CPU cores, GPU cores, and Neural Engine—at full bandwidth. When a computation produces results on the GPU, those results are immediately available to the CPU without any copying.

The implications for interpretability research are profound. Capturing activations no longer requires expensive GPU-to-CPU transfers. An activation tensor produced by a transformer layer can be analyzed by CPU code, visualized in a plotting library, and saved to disk, all without ever being copied. The unified memory pool can also be much larger than discrete GPU memory—current configurations support up to 192GB, far exceeding the 80GB available on even the most expensive data center GPUs.

### 3.2 Quantitative Analysis

To appreciate the practical impact, consider the memory requirements for interpretability work on a 7B parameter model:

- Model parameters (4-bit quantized): ~3.5GB
- Key-value cache for 2048 tokens: ~1GB
- Intermediate activations (all layers): ~2-4GB per forward pass
- Analysis workspace: Variable, often 10-50GB for batch analysis

On a traditional system with 24GB GPU memory, researchers must carefully manage what fits in GPU memory, often forcing compromises between model size, sequence length, and activation caching. Running out of memory mid-experiment is a common frustration.

On an Apple Silicon system with 64GB or more of unified memory, these constraints largely disappear. The entire workflow—model inference, activation capture, statistical analysis, visualization—operates within a single coherent memory space. More importantly, the absence of memory copies means that capturing activations has minimal overhead compared to simply running inference.

### 3.3 Economic Considerations

Beyond technical advantages, unified memory architecture has significant economic implications for research accessibility. Cloud GPU instances with sufficient memory for serious interpretability work cost $2-8 per hour, depending on provider and GPU type. A researcher running experiments for 8 hours daily would face monthly costs of $400-$1,600—a substantial burden for unfunded researchers, students, or those at institutions without cloud computing budgets.

Apple Silicon hardware, while requiring upfront investment, has no ongoing costs. A MacBook Pro or Mac Studio, once purchased, can run unlimited interpretability experiments. For researchers who expect to work in this area for years, the economics strongly favor local hardware.

Moreover, cloud computing introduces friction that impedes the kind of rapid iteration that drives scientific progress. Starting a cloud instance, waiting for model loading, and managing session state all add overhead that discourages exploratory analysis. Local computing enables immediate experimentation—a researcher can have a new idea and be testing it within seconds.

---

## 4. Design Philosophy: Learning from Existing Tools

### 4.1 The Interpretability Tool Landscape

Several excellent interpretability libraries have been developed for PyTorch, most notably TransformerLens (Nanda, 2022) and nnsight (Fiotto-Kaufman et al., 2024). These tools have demonstrated the value of well-designed abstractions for interpretability research, and mlxterp draws heavily on their insights.

TransformerLens pioneered the idea of "hooking" into transformer computations, providing clean access to attention patterns, residual stream contributions, and other internals. Its comprehensive API has enabled much of the field's empirical work. However, TransformerLens requires model-specific implementations—each architecture needs its own "hooked" version—limiting its applicability to a curated set of models.

nnsight took a different approach, using Python's meta-programming capabilities to wrap arbitrary PyTorch models without requiring model-specific code. This "model-agnostic" design philosophy means nnsight can work with any PyTorch model, dramatically expanding its applicability. nnsight also pioneered the concept of remote execution, allowing researchers to run interpretability experiments on models hosted by infrastructure providers like NDIF (National Deep Inference Fabric).

### 4.2 Design Decisions in mlxterp

mlxterp synthesizes insights from both approaches while adapting to MLX's programming model. Our key design decisions include:

**Model-Agnostic Wrapping**: Like nnsight, mlxterp works with arbitrary MLX models through recursive module discovery. When you wrap a model with `InterpretableModel`, mlxterp traverses the module tree, identifies all computational components, and installs transparent wrappers that capture activations and enable interventions. No model-specific code is required.

**Context Manager Tracing**: Both nnsight and mlxterp use Python context managers to delineate the scope of tracing operations. This pattern makes code intent explicit—everything inside the `with model.trace():` block is subject to tracing—and ensures proper cleanup of temporary state.

**Lazy Evaluation Integration**: MLX, like JAX, uses lazy evaluation—computations are not executed until their results are needed. mlxterp embraces this paradigm rather than fighting it. When you call `.save()` on an activation, you're not immediately capturing data; you're marking that activation for capture when the computation graph eventually executes.

**API Compatibility**: We deliberately designed mlxterp's API to feel familiar to nnsight users. This isn't merely about reducing learning curves—it's about enabling a research workflow where techniques developed locally on Apple Silicon can be directly applied to larger models via nnsight's remote execution capabilities.

### 4.3 The Local-to-Cloud Research Pathway

This last point deserves elaboration. Mechanistic interpretability research typically proceeds through stages:

1. **Hypothesis Formation**: Initial ideas emerge from intuition, prior work, or unexpected observations
2. **Rapid Prototyping**: Quick experiments test whether an idea has merit
3. **Careful Validation**: Promising findings are validated with more rigorous methodology
4. **Scaling Verification**: Results are confirmed on larger, more capable models

Different stages have different computational requirements. Hypothesis formation and rapid prototyping benefit from immediate feedback and low friction—exactly what local computing provides. Careful validation may require more examples or longer sequences, potentially exceeding local resources. Scaling verification definitionally requires access to larger models than can run locally.

By designing mlxterp's API to mirror nnsight's, we enable researchers to move fluidly between stages. A technique developed and debugged locally can be applied to frontier models via NDIF (ndif.us) or EDIF (edif.ai) with minimal code changes. The same mental models, the same patterns of investigation, transfer across the local-cloud boundary.

This design also future-proofs research workflows. As Apple Silicon capabilities increase (current M4 Max systems support 128GB unified memory; future systems will likely support more), the range of models analyzable locally will expand. Research techniques developed today will continue working as hardware improves.

---

## 5. Technical Implementation

### 5.1 Activation Capture Architecture

At its core, mlxterp implements a transparent interposition layer between model components and their callers. When a model is wrapped with `InterpretableModel`, each submodule is wrapped with a lightweight proxy that:

1. Intercepts calls to the original module
2. Invokes the original computation
3. Stores the output in a shared activation dictionary
4. Returns the output (potentially modified by interventions) to the caller

This design has several important properties. First, it preserves the original model's behavior exactly when no interventions are active—the proxy simply passes through computations. Second, it captures activations at the granularity of individual modules, providing access to attention outputs, MLP intermediates, and layer normalizations separately. Third, it imposes minimal overhead, as the interposition layer adds only dictionary storage operations to the critical path.

The activation dictionary is keyed by the full path to each module (e.g., `model.model.layers.5.self_attn.q_proj`), providing a systematic naming scheme that works across different model architectures. For a typical transformer model, this results in approximately 196 captured activations per forward pass—far more than most research requires, but available when fine-grained analysis is needed.

### 5.2 The Intervention System

Interventions—modifications to activations during the forward pass—are central to causal interpretability methods. mlxterp provides a composable intervention system that supports common operations while remaining extensible.

Built-in interventions include:

- **Zeroing**: Set activations to zero, removing a component's contribution
- **Scaling**: Multiply activations by a constant, attenuating or amplifying effects
- **Addition**: Add a vector to activations, implementing steering or ablation
- **Replacement**: Replace activations entirely, implementing activation patching
- **Clamping**: Restrict activations to a range, testing saturation effects
- **Noise**: Add Gaussian noise, testing robustness to perturbation

These interventions can be composed, allowing complex experimental manipulations to be expressed concisely. A researcher might scale an activation by 0.5, add noise, and then clamp the result—all in a single composed intervention.

For cases where built-in interventions are insufficient, arbitrary Python functions can serve as interventions. Any function that accepts an activation tensor and returns a modified tensor can be used, enabling experimentation with novel manipulation strategies.

### 5.3 Analysis Methods

Beyond infrastructure for activation capture and intervention, mlxterp provides implementations of common interpretability analyses:

**Logit Lens** (nostalgebraist, 2020): Projects intermediate activations to vocabulary space by applying the model's unembedding matrix. This reveals what "prediction" each layer would make if it were the final layer, showing how the model builds up its answer through successive refinements. mlxterp's implementation handles the complexity of weight-tied embeddings and final layer normalization automatically.

**Tuned Lens** (Belrose et al., 2023): An improvement on the logit lens that learns a small affine transformation for each layer to correct for coordinate system differences between layers. This produces more accurate intermediate predictions, particularly for early layers. mlxterp includes both training code and inference support for tuned lens analysis.

**Activation Patching**: A systematic method for identifying which components are causally important for a particular behavior. mlxterp's implementation automates the process of running clean and corrupted inputs, patching activations, and measuring recovery, producing results that can be immediately visualized.

---

## 6. Practical Applications

To ground these abstractions in concrete research practice, we present several example applications that illustrate mlxterp's capabilities.

### 6.1 Investigating Factual Recall

A fundamental question in language model interpretability is how models store and retrieve factual knowledge. When a model correctly completes "The capital of France is" with "Paris," where does this knowledge come from? Is it distributed across many layers, or localized in specific components?

Using mlxterp, we can investigate this question through activation patching:

```python
from mlxterp import InterpretableModel

model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

# Test which MLP layers are important for factual recall
results = model.activation_patching(
    clean_text="The capital of France is Paris",
    corrupted_text="The capital of France is London",
    component="mlp",
    plot=True
)

# Identify the most important layers
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for layer_idx, recovery in sorted_results[:5]:
    print(f"Layer {layer_idx}: {recovery:.1f}% recovery")
```

This experiment reveals which layers are most critical for producing the correct factual completion. High recovery percentages indicate layers where the model's "knowledge" of Paris being France's capital is concentrated. Research using similar methodology has shown that factual knowledge often localizes in middle layers' MLP components, consistent with theories that MLPs serve as key-value memories (Geva et al., 2021).

### 6.2 Tracing Prediction Evolution

The logit lens technique allows us to trace how a model's predictions evolve through its layers:

```python
# Analyze prediction evolution
results = model.logit_lens(
    "The Eiffel Tower is located in the city of",
    layers=list(range(16)),
    plot=True
)

# Print predictions at the final token position
for layer_idx in range(0, 16, 4):
    prediction = results[layer_idx][-1][0][2]  # Top prediction at last position
    print(f"Layer {layer_idx}: predicts '{prediction}'")
```

This analysis typically reveals a characteristic pattern: early layers make generic predictions (common words like "the" or "a"), middle layers begin to incorporate context (perhaps "Paris" or "France"), and late layers converge on the correct answer with high confidence. Deviations from this pattern can indicate interesting model behaviors worth investigating.

### 6.3 Developing Steering Vectors

Activation steering has emerged as a promising technique for controlling model behavior without fine-tuning. The basic approach computes a "steering vector" as the difference between activations for contrasting prompts, then adds this vector during inference to bias the model toward desired behaviors.

```python
import mlx.core as mx
from mlxterp import interventions as iv

# Compute steering vector from contrastive examples
with model.trace("I think this is absolutely wonderful and amazing") as positive:
    pos_activation = positive.activations['model.model.layers.10']

with model.trace("I think this is terrible and disappointing") as negative:
    neg_activation = negative.activations['model.model.layers.10']

# Compute difference as steering vector
sentiment_vector = pos_activation.mean(axis=1) - neg_activation.mean(axis=1)

# Apply steering to a neutral prompt
with model.trace(
    "The movie was",
    interventions={'model.model.layers.10': iv.add_vector(sentiment_vector * 2.0)}
) as steered:
    output_logits = steered.activations['__model_output__']
```

mlxterp's low-overhead activation capture makes it practical to iterate rapidly on steering vector development, testing different layers, scaling factors, and contrastive example pairs.

---

## 7. Relationship to Remote Inference Infrastructure

### 7.1 The NDIF and EDIF Ecosystem

The interpretability research community has developed shared infrastructure for accessing large models. The National Deep Inference Fabric (NDIF) provides U.S. researchers with free access to frontier models for academic research. The European Deep Inference Fabric (EDIF) serves a similar role for European researchers. Both services use nnsight's remote execution capabilities, allowing researchers to run interpretability experiments on models too large for local deployment.

These services represent a significant democratization of access to frontier model internals. A researcher at a small institution can now run activation patching experiments on 70B parameter models without any GPU infrastructure—the computation happens on NDIF's servers, with only the results returned to the researcher.

### 7.2 Complementary Workflows

mlxterp and remote inference services are complements, not substitutes. Local computing excels for rapid iteration, debugging, and exploratory analysis—situations where minimizing latency between idea and result maximizes productivity. Remote computing enables scaling to models beyond local capacity and provides access to frontier capabilities that may not have efficient local implementations.

The ideal workflow leverages both: develop and refine techniques locally on smaller models using mlxterp, then validate findings on larger models via nnsight and NDIF. Because the two systems share API patterns, this transition is smooth—the same mental models and much of the same code transfer across the boundary.

This complementarity also provides a natural growth path for researchers. Someone new to interpretability can begin with purely local work using mlxterp, building intuition and developing techniques. As they tackle more ambitious questions requiring larger models, they can gradually incorporate remote resources without abandoning their local development workflow.

---

## 8. Current Limitations and Future Directions

### 8.1 Limitations

We acknowledge several limitations of the current mlxterp implementation:

**Model Size Constraints**: While Apple Silicon's unified memory supports larger models than discrete GPUs, there are still limits. A 70B parameter model, even with aggressive quantization, requires more memory than current consumer hardware provides. Truly frontier-scale research still requires cloud resources.

**Ecosystem Maturity**: The MLX ecosystem is younger than PyTorch's, with fewer pre-trained models, less community tooling, and ongoing API evolution. Researchers may need to convert models from other formats or work around missing features.

**Performance Optimization**: While unified memory eliminates copy overhead, MLX's computational performance is still maturing. Some operations may be slower than optimized CUDA implementations, though for interpretability work (where activation capture dominates runtime), this is often acceptable.

### 8.2 Planned Developments

We are actively developing several extensions to mlxterp:

**Sparse Autoencoder Support**: SAEs have become central to modern interpretability research, revealing interpretable features in model activations. We are implementing SAE training on captured activations and compatibility with existing SAE formats (particularly SAELens), enabling feature-based analysis workflows.

**Attention Pattern Analysis**: Built-in tools for analyzing attention patterns, including visualization, pattern matching across examples, and statistical characterization of attention head behaviors.

**Circuit Extraction Tools**: Semi-automated tools for identifying and extracting circuits, building on activation patching to provide higher-level abstractions for circuit discovery.

**Enhanced NDIF Integration**: Tighter integration with nnsight for seamless local-to-remote transitions, potentially including unified result formats and analysis tools that work across both execution modes.

### 8.3 Long-Term Vision

Looking further ahead, we envision interpretability tooling that enables qualitatively new research paradigms:

**Real-Time Interpretability**: Analyzing model behavior during generation, with interventions that adapt based on intermediate computations. This could enable dynamic steering that responds to detected model states.

**Interpretability-Informed Training**: Using insights from interpretability research to guide model development, potentially training models that are more interpretable by design.

**Automated Hypothesis Generation**: Tools that not only test hypotheses but help generate them, identifying anomalous patterns that warrant investigation.

These directions require not just better tools but better scientific understanding of neural network computation. We hope that by lowering barriers to interpretability research, mlxterp will contribute to the diverse research community needed to achieve this understanding.

---

## 9. Conclusion

Mechanistic interpretability represents one of our best hopes for understanding and ensuring the safety of advanced AI systems. Yet the field's progress has been limited by computational barriers that exclude many potential contributors. The combination of Apple Silicon's unified memory architecture and thoughtfully designed software tools like mlxterp offers a path to broader participation.

We have presented mlxterp not merely as a technical artifact but as part of a broader vision for democratized interpretability research. By enabling serious interpretability work on consumer hardware, by designing APIs that facilitate transitions to cloud resources when needed, and by contributing to an ecosystem of compatible tools, we hope to accelerate progress on questions that matter enormously for humanity's future with AI.

The challenges ahead are significant. We do not fully understand how neural networks implement the capabilities they exhibit, how to reliably detect undesirable behaviors, or how to align AI systems with human values. These are not problems that any single tool or research group will solve. They require the collective effort of a large, diverse community of researchers bringing different perspectives and approaches.

mlxterp is our contribution to building that community. We invite researchers to use these tools, extend them, and share what they learn. The source code is available at github.com/coairesearch/mlxterp, and we welcome contributions of all kinds.

Understanding the systems we build is not optional—it is a prerequisite for building systems we can trust. We hope mlxterp makes that understanding a little more accessible.

---

## References

Belrose, N., Furman, Z., Smith, L., Halawi, D., Ostrovsky, I., McKinney, L., ... & Steinhardt, J. (2023). Eliciting latent predictions from transformers with the tuned lens. *arXiv preprint arXiv:2303.08112*.

Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., ... & Olah, C. (2023). Towards monosemanticity: Decomposing language models with dictionary learning. *Transformer Circuits Thread*.

Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). Sparse autoencoders find highly interpretable features in language models. *arXiv preprint arXiv:2309.08600*.

Fiotto-Kaufman, J., Loftus, A., Todd, E., Brinkmann, J., Juang, C., Pal, K., ... & Bau, D. (2024). NNsight and NDIF: Democratizing access to foundation model internals. *arXiv preprint arXiv:2407.14561*.

Geva, M., Schuster, R., Berant, J., & Levy, O. (2021). Transformer feed-forward layers are key-value memories. *arXiv preprint arXiv:2012.14913*.

Goldowsky-Dill, N., MacLeod, C., Sato, L., & Arora, A. (2023). Localizing model behavior with path patching. *arXiv preprint arXiv:2304.05969*.

Nanda, N. (2022). TransformerLens. GitHub repository. https://github.com/neelnanda-io/TransformerLens

nostalgebraist. (2020). interpreting GPT: the logit lens. *LessWrong*.

Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., ... & Olah, C. (2022). In-context learning and induction heads. *Transformer Circuits Thread*.

Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Singer, Y., & Shieber, S. (2020). Investigating gender bias in language models using causal mediation analysis. *Advances in Neural Information Processing Systems*.

Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2022). Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. *arXiv preprint arXiv:2211.00593*.

---

## Acknowledgments

We thank the developers of nnsight, TransformerLens, and the broader interpretability research community for their foundational work that mlxterp builds upon. We also thank Apple's MLX team for creating a framework that makes this work possible.

---

## Citation

```bibtex
@software{mlxterp2025,
  title = {mlxterp: Mechanistic Interpretability for MLX on Apple Silicon},
  author = {Schacht, Sigurd and {COAI Research Institute}},
  year = {2025},
  url = {https://github.com/coairesearch/mlxterp},
  note = {Open source library for neural network interpretability research}
}
```

---

*For questions, feedback, or collaboration inquiries, please visit [coairesearch.org](https://coairesearch.org) or open an issue on our [GitHub repository](https://github.com/coairesearch/mlxterp).*
