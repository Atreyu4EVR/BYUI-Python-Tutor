import streamlit as st

st.subheader("Model Card: Meta-Llama 3.1 8B-Instruct")
st.markdown(
    """
    ## Model Information

    The Meta Llama 3.1 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction tuned generative models in 8B, 70B, and 405B sizes (text in/text out). The Llama 3.1 instruction tuned text-only models (8B, 70B, 405B) are optimized for multilingual dialogue use cases and outperform many of the available open-source and closed chat models on common industry benchmarks.

    **Model developer:** Meta

    **Model Architecture:** Llama 3.1 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety.
    """
)

st.markdown(
    """
    <table>
      <tr>
       <td></td>
       <td><strong>Training Data</strong></td>
       <td><strong>Params</strong></td>
       <td><strong>Input modalities</strong></td>
       <td><strong>Output modalities</strong></td>
       <td><strong>Context length</strong></td>
       <td><strong>GQA</strong></td>
       <td><strong>Token count</strong></td>
       <td><strong>Knowledge cutoff</strong></td>
      </tr>
      <tr>
       <td rowspan="3">Llama 3.1 (text only)</td>
       <td rowspan="3">A new mix of publicly available online data.</td>
       <td>8B</td>
       <td>Multilingual Text</td>
       <td>Multilingual Text and code</td>
       <td>128k</td>
       <td>Yes</td>
       <td rowspan="3">15T+</td>
       <td rowspan="3">December 2023</td>
      </tr>
      <tr>
       <td>70B</td>
       <td>Multilingual Text</td>
       <td>Multilingual Text and code</td>
       <td>128k</td>
       <td>Yes</td>
      </tr>
      <tr>
       <td>405B</td>
       <td>Multilingual Text</td>
       <td>Multilingual Text and code</td>
       <td>128k</td>
       <td>Yes</td>
      </tr>
    </table>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    **Supported languages:** English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai.

    **Llama 3.1 family of models**. Token counts refer to pretraining data only. All model versions use Grouped-Query Attention (GQA) for improved inference scalability.

    **Model Release Date:** July 23, 2024.

    **Status:** This is a static model trained on an offline dataset. Future versions of the tuned models will be released as we improve model safety with community feedback.

    **License:** A custom commercial license, the Llama 3.1 Community License, is available at: [https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE)

    Where to send questions or comments about the model Instructions on how to provide feedback or comments on the model can be found in the model [README](https://github.com/meta-llama/llama3). For more technical information about generation parameters and recipes for how to use Llama 3.1 in applications, please go [here](https://github.com/meta-llama/llama-recipes).
    """
)

st.markdown(
    """
    ## Intended Use

    **Intended Use Cases** Llama 3.1 is intended for commercial and research use in multiple languages. Instruction tuned text-only models are intended for assistant-like chat, whereas pretrained models can be adapted for a variety of natural language generation tasks. The Llama 3.1 model collection also supports the ability to leverage the outputs of its models to improve other models including synthetic data generation and distillation. The Llama 3.1 Community License allows for these use cases.

    **Out-of-scope** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in any other way that is prohibited by the Acceptable Use Policy and Llama 3.1 Community License. Use in languages beyond those explicitly referenced as supported in this model card.

    **<span style="text-decoration:underline;">Note</span>:** Llama 3.1 has been trained on a broader collection of languages than the 8 supported languages. Developers may fine-tune Llama 3.1 models for languages beyond the 8 supported languages provided they comply with the Llama 3.1 Community License and the Acceptable Use Policy and in such cases are responsible for ensuring that any uses of Llama 3.1 in additional languages is done in a safe and responsible manner.
    """, unsafe_allow_html=True
)

st.markdown(
    """
    ## Hardware and Software

    **Training Factors** We used custom training libraries, Meta's custom-built GPU cluster, and production infrastructure for pretraining. Fine-tuning, annotation, and evaluation were also performed on production infrastructure.

    **Training Energy Use** Training utilized a cumulative of **39.3M** GPU hours of computation on H100-80GB (TDP of 700W) type hardware, per the table below. Training time is the total GPU time required for training each model and power consumption is the peak power capacity per GPU device used, adjusted for power usage efficiency.

    **Training Greenhouse Gas Emissions** Estimated total location-based greenhouse gas emissions were **11,390** tons CO2eq for training. Since 2020, Meta has maintained net zero greenhouse gas emissions in its global operations and matched 100% of its electricity use with renewable energy, therefore the total market-based greenhouse gas emissions for training were 0 tons CO2eq.
    """
)

st.markdown(
    """
    <table>
      <tr>
       <td></td>
       <td><strong>Training Time (GPU hours)</strong></td>
       <td><strong>Training Power Consumption (W)</strong></td>
       <td><strong>Training Location-Based Greenhouse Gas Emissions</strong><p><strong>(tons CO2eq)</strong></td>
       <td><strong>Training Market-Based Greenhouse Gas Emissions</strong><p><strong>(tons CO2eq)</strong></td>
      </tr>
      <tr>
       <td>Llama 3.1 8B</td>
       <td>1.46M</td>
       <td>700</td>
       <td>420</td>
       <td>0</td>
      </tr>
      <tr>
       <td>Llama 3.1 70B</td>
       <td>7.0M</td>
       <td>700</td>
       <td>2,040</td>
       <td>0</td>
      </tr>
      <tr>
       <td>Llama 3.1 405B</td>
       <td>30.84M</td>
       <td>700</td>
       <td>8,930</td>
       <td>0</td>
      </tr>
      <tr>
       <td>Total</td>
       <td>39.3M</td>
       <td>11,390</td>
       <td>0</td>
      </tr>
    </table>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    The methodology used to determine training energy use and greenhouse gas emissions can be found [here](https://arxiv.org/pdf/2204.05149). Since Meta is openly releasing these models, the training energy use and greenhouse gas emissions will not be incurred by others.
    """
)

st.markdown(
    """
    ## Training Data

    **Overview:** Llama 3.1 was pretrained on ~15 trillion tokens of data from publicly available sources. The fine-tuning data includes publicly available instruction datasets, as well as over 25M synthetically generated examples.

    **Data Freshness:** The pretraining data has a cutoff of December 2023.
    """
)

st.subheader("Model Card: Google Gemma 1.1 7B-Instruct")
st.markdown(
    """
        # Gemma Model Card
    
    **Model Page**: [Gemma](https://ai.google.dev/gemma/docs)
    
    This model card corresponds to the latest 7B instruct version of the Gemma model. Here you can find other models in the Gemma family:
    
    |    | Base                                               | Instruct                                                             |
    |----|----------------------------------------------------|----------------------------------------------------------------------|
    | 2B | [gemma-2b](https://huggingface.co/google/gemma-2b) | [gemma-1.1-2b-it](https://huggingface.co/google/gemma-1.1-2b-it) |
    | 7B | [gemma-7b](https://huggingface.co/google/gemma-7b) | [**gemma-1.1-7b-it**](https://huggingface.co/google/gemma-1.1-7b-it)     |
    
    **Release Notes**
    
    This is Gemma 1.1 7B (IT), an update over the original instruction-tuned Gemma release.
    
    Gemma 1.1 was trained using a novel RLHF method, leading to substantial gains on quality, coding capabilities, factuality, instruction following and multi-turn conversation quality. We also fixed a bug in multi-turn conversations, and made sure that model responses don't always start with `"Sure,"`.
    
    We believe this release represents an improvement for most use cases, but we encourage users to test in their particular applications. The previous model [will continue to be available in the same repo](https://huggingface.co/google/gemma-7b-it). We appreciate the enthusiastic adoption of Gemma, and we continue to welcome all feedback from the community.
    
    **Resources and Technical Documentation**:
    
    * [Responsible Generative AI Toolkit](https://ai.google.dev/responsible)
    * [Gemma on Kaggle](https://www.kaggle.com/models/google/gemma)
    * [Gemma on Vertex Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/335)
    
    **Terms of Use**: [Terms](https://www.kaggle.com/models/google/gemma/license/consent/verify/huggingface?returnModelRepoId=google/gemma-1.1-7b-it)
    
    **Authors**: Google
    
    ## Model Information
    
    Summary description and brief definition of inputs and outputs.
    
    ### Description
    
    Gemma is a family of lightweight, state-of-the-art open models from Google,
    built from the same research and technology used to create the Gemini models.
    They are text-to-text, decoder-only large language models, available in English,
    with open weights, pre-trained variants, and instruction-tuned variants. Gemma
    models are well-suited for a variety of text generation tasks, including
    question answering, summarization, and reasoning. Their relatively small size
    makes it possible to deploy them in environments with limited resources such as
    a laptop, desktop or your own cloud infrastructure, democratizing access to
    state of the art AI models and helping foster innovation for everyone.
    
    ### Usage
    
    Below we share some code snippets on how to get quickly started with running the model. First make sure to `pip install -U transformers`, then copy the snippet from the section that is relevant for your usecase.
    
    #### Running the model on a CPU
    
    As explained below, we recommend `torch.bfloat16` as the default dtype. You can use [a different precision](#precisions) if necessary.
    
    ```python
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-7b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-1.1-7b-it",
        torch_dtype=torch.bfloat16
    )
    
    input_text = "Write me a poem about Machine Learning."
    input_ids = tokenizer(input_text, return_tensors="pt")
    
    outputs = model.generate(**input_ids, max_new_tokens=50)
    print(tokenizer.decode(outputs[0]))
    ```
    
    #### Running the model on a single / multi GPU
    
    
    ```python
    # pip install accelerate
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-7b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-1.1-7b-it",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    input_text = "Write me a poem about Machine Learning."
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))
    ```
    
    <a name="precisions"></a>
    #### Running the model on a GPU using different precisions
    
    The native weights of this model were exported in `bfloat16` precision. You can use `float16`, which may be faster on certain hardware, indicating the `torch_dtype` when loading the model. For convenience, the `float16` revision of the repo contains a copy of the weights already converted to that precision.
    
    You can also use `float32` if you skip the dtype, but no precision increase will occur (model weights will just be upcasted to `float32`). See examples below.
    
    * _Using `torch.float16`_
    
    ```python
    # pip install accelerate
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-7b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-1.1-7b-it",
        device_map="auto",
        torch_dtype=torch.float16,
        revision="float16",
    )
    
    input_text = "Write me a poem about Machine Learning."
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))
    ```
    
    * _Using `torch.bfloat16`_
    
    ```python
    # pip install accelerate
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-1.1-7b-it",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    input_text = "Write me a poem about Machine Learning."
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))
    ```
    
    * _Upcasting to `torch.float32`_
    
    ```python
    # pip install accelerate
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-7b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-1.1-7b-it",
        device_map="auto"
    )
    
    input_text = "Write me a poem about Machine Learning."
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))
    ```
    
    #### Quantized Versions through `bitsandbytes`
    
    * _Using 8-bit precision (int8)_
    
    ```python
    # pip install bitsandbytes accelerate
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-7b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-1.1-7b-it",
        quantization_config=quantization_config
    )
    
    input_text = "Write me a poem about Machine Learning."
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))
    ```
    
    * _Using 4-bit precision_
    
    ```python
    # pip install bitsandbytes accelerate
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-7b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-1.1-7b-it",
        quantization_config=quantization_config
    )
    
    input_text = "Write me a poem about Machine Learning."
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))
    ```
    
    
    #### Other optimizations
    
    * _Flash Attention 2_
    
    First make sure to install `flash-attn` in your environment `pip install flash-attn`
    
    ```diff
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
    +   attn_implementation="flash_attention_2"
    ).to(0)
    ```
    
    #### Running the model in JAX / Flax
    
    Use the `flax` branch of the repository:
    
    ```python
    import jax.numpy as jnp
    from transformers import AutoTokenizer, FlaxGemmaForCausalLM
    
    model_id = "google/gemma-1.1-7b-it"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    
    model, params = FlaxGemmaForCausalLM.from_pretrained(
    		model_id,
    		dtype=jnp.bfloat16,
    		revision="flax",
    		_do_init=False,
    )
    
    inputs = tokenizer("Valencia and Málaga are", return_tensors="np", padding=True)
    output = model.generate(**inputs, params=params, max_new_tokens=20, do_sample=False)
    output_text = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
    ```
    
    [Check this notebook](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/jax_gemma.ipynb) for a comprehensive walkthrough on how to parallelize JAX inference.
    
    
    ### Chat Template
    
    The instruction-tuned models use a chat template that must be adhered to for conversational use.
    The easiest way to apply it is using the tokenizer's built-in chat template, as shown in the following snippet.
    
    Let's load the model and apply the chat template to a conversation. In this example, we'll start with a single user interaction:
    
    ```py
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
    import torch
    
    model_id = "google/gemma-1.1-7b-it"
    dtype = torch.bfloat16
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype=dtype,
    )
    
    chat = [
        { "role": "user", "content": "Write a hello world program" },
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    ```
    
    At this point, the prompt contains the following text:
    
    ```
    <bos><start_of_turn>user
    Write a hello world program<end_of_turn>
    <start_of_turn>model
    ```
    
    As you can see, each turn is preceded by a `<start_of_turn>` delimiter and then the role of the entity
    (either `user`, for content supplied by the user, or `model` for LLM responses). Turns finish with
    the `<end_of_turn>` token.
    
    You can follow this format to build the prompt manually, if you need to do it without the tokenizer's
    chat template.
    
    After the prompt is ready, generation can be performed like this:
    
    ```py
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)
    ```
    
    ### Fine-tuning
    
    You can find some fine-tuning scripts under the [`examples/` directory](https://huggingface.co/google/gemma-7b/tree/main/examples) of [`google/gemma-7b`](https://huggingface.co/google/gemma-7b) repository. To adapt them to this model, simply change the model-id to `google/gemma-1.1-7b-it`.
    
    We provide:
    
    * A script to perform Supervised Fine-Tuning (SFT) on UltraChat dataset using QLoRA
    * A script to perform SFT using FSDP on TPU devices
    * A notebook that you can run on a free-tier Google Colab instance to perform SFT on the English quotes dataset
    
    ### Inputs and outputs
    
    *   **Input:** Text string, such as a question, a prompt, or a document to be
        summarized.
    *   **Output:** Generated English-language text in response to the input, such
        as an answer to a question, or a summary of a document.
    
    ## Model Data
    
    Data used for model training and how the data was processed.
    
    ### Training Dataset
    
    These models were trained on a dataset of text data that includes a wide variety
    of sources, totaling 6 trillion tokens. Here are the key components:
    
    * Web Documents: A diverse collection of web text ensures the model is exposed
      to a broad range of linguistic styles, topics, and vocabulary. Primarily
      English-language content.
    * Code: Exposing the model to code helps it to learn the syntax and patterns of
      programming languages, which improves its ability to generate code or
      understand code-related questions.
    * Mathematics: Training on mathematical text helps the model learn logical
      reasoning, symbolic representation, and to address mathematical queries.
    
    The combination of these diverse data sources is crucial for training a powerful
    language model that can handle a wide variety of different tasks and text
    formats.
    
    ### Data Preprocessing
    
    Here are the key data cleaning and filtering methods applied to the training
    data:
    
    * CSAM Filtering: Rigorous CSAM (Child Sexual Abuse Material) filtering was
      applied at multiple stages in the data preparation process to ensure the
      exclusion of harmful and illegal content
    * Sensitive Data Filtering: As part of making Gemma pre-trained models safe and
      reliable, automated techniques were used to filter out certain personal
      information and other sensitive data from training sets.
    * Additional methods: Filtering based on content quality and safely in line with
      [our policies](https://storage.googleapis.com/gweb-uniblog-publish-prod/documents/2023_Google_AI_Principles_Progress_Update.pdf#page=11).
    
    ## Implementation Information
    
    Details about the model internals.
    
    ### Hardware
    
    Gemma was trained using the latest generation of
    [Tensor Processing Unit (TPU)](https://cloud.google.com/tpu/docs/intro-to-tpu) hardware (TPUv5e).
    
    Training large language models requires significant computational power. TPUs,
    designed specifically for matrix operations common in machine learning, offer
    several advantages in this domain:
    
    * Performance: TPUs are specifically designed to handle the massive computations
      involved in training LLMs. They can speed up training considerably compared to
      CPUs.
    * Memory: TPUs often come with large amounts of high-bandwidth memory, allowing
      for the handling of large models and batch sizes during training. This can
      lead to better model quality.
    * Scalability: TPU Pods (large clusters of TPUs) provide a scalable solution for
      handling the growing complexity of large foundation models. You can distribute
      training across multiple TPU devices for faster and more efficient processing.
    * Cost-effectiveness: In many scenarios, TPUs can provide a more cost-effective
      solution for training large models compared to CPU-based infrastructure,
      especially when considering the time and resources saved due to faster
      training.
    * These advantages are aligned with
      [Google's commitments to operate sustainably](https://sustainability.google/operating-sustainably/).
    
    ### Software
    
    Training was done using [JAX](https://github.com/google/jax) and [ML Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/ml-pathways).
    
    JAX allows researchers to take advantage of the latest generation of hardware,
    including TPUs, for faster and more efficient training of large models.
    
    ML Pathways is Google's latest effort to build artificially intelligent systems
    capable of generalizing across multiple tasks. This is specially suitable for
    [foundation models](https://ai.google/discover/foundation-models/), including large language models like
    these ones.
    
    Together, JAX and ML Pathways are used as described in the
    [paper about the Gemini family of models](https://arxiv.org/abs/2312.11805); "the 'single
    controller' programming model of Jax and Pathways allows a single Python
    process to orchestrate the entire training run, dramatically simplifying the
    development workflow."
    
    ## Evaluation
    
    Model evaluation metrics and results.
    
    ### Benchmark Results
    
    The pre-trained base models were evaluated against a large collection of different datasets and
    metrics to cover different aspects of text generation:
    
    | Benchmark                      | Metric        | 2B Params | 7B Params |
    | ------------------------------ | ------------- | ----------- | --------- |
    | [MMLU](https://arxiv.org/abs/2009.03300)                   | 5-shot, top-1 | 42.3        | 64.3      |
    | [HellaSwag](https://arxiv.org/abs/1905.07830)         | 0-shot        |71.4        | 81.2      |
    | [PIQA](https://arxiv.org/abs/1911.11641)                   | 0-shot        | 77.3        | 81.2      |
    | [SocialIQA](https://arxiv.org/abs/1904.09728)      | 0-shot        | 49.7        | 51.8      |
    | [BooIQ](https://arxiv.org/abs/1905.10044)                | 0-shot        | 69.4        | 83.2      |
    | [WinoGrande](https://arxiv.org/abs/1907.10641)       | partial score | 65.4        | 72.3      |
    | [CommonsenseQA](https://arxiv.org/abs/1811.00937) | 7-shot        | 65.3        | 71.3      |
    | [OpenBookQA](https://arxiv.org/abs/1809.02789)       |               | 47.8        | 52.8      |
    | [ARC-e](https://arxiv.org/abs/1911.01547)                  |               | 73.2        | 81.5      |
    | [ARC-c](https://arxiv.org/abs/1911.01547)                   |               | 42.1        | 53.2      |
    | [TriviaQA](https://arxiv.org/abs/1705.03551)           | 5-shot        | 53.2        | 63.4      |
    | [Natural Questions](https://github.com/google-research-datasets/natural-questions)  | 5-shot        | 12.5       | 23        |
    | [HumanEval](https://arxiv.org/abs/2107.03374)      | pass@1        | 22.0        | 32.3      |
    | [MBPP](https://arxiv.org/abs/2108.07732)                   | 3-shot        | 29.2        | 44.4      |
    | [GSM8K](https://arxiv.org/abs/2110.14168)                | maj@1         | 17.7        | 46.4      |
    | [MATH](https://arxiv.org/abs/2108.07732)                   | 4-shot        | 11.8          | 24.3      |
    | [AGIEval](https://arxiv.org/abs/2304.06364)           |               | 24.2        | 41.7      |
    | [BIG-Bench](https://arxiv.org/abs/2206.04615)         |               | 35.2        | 55.1      |
    | ------------------------------ | ------------- | ----------- | --------- |
    | **Average**                    |               | **45.0**    | **56.9**  |
    
    ## Ethics and Safety
    
    Ethics and safety evaluation approach and results.
    
    ### Evaluation Approach
    
    Our evaluation methods include structured evaluations and internal red-teaming
    testing of relevant content policies. Red-teaming was conducted by a number of
    different teams, each with different goals and human evaluation metrics. These
    models were evaluated against a number of different categories relevant to
    ethics and safety, including:
    
    * Text-to-Text Content Safety: Human evaluation on prompts covering safety
      policies including child sexual abuse and exploitation, harassment, violence
      and gore, and hate speech.
    * Text-to-Text Representational Harms: Benchmark against relevant academic
      datasets such as [WinoBias](https://arxiv.org/abs/1804.06876) and [BBQ Dataset](https://arxiv.org/abs/2110.08193v2).
    * Memorization: Automated evaluation of memorization of training data, including
      the risk of personally identifiable information exposure.
    * Large-scale harm: Tests for "dangerous capabilities," such as chemical,
      biological, radiological, and nuclear (CBRN) risks.
    
    ### Evaluation Results
    
    The results of ethics and safety evaluations are within acceptable thresholds
    for meeting [internal policies](https://storage.googleapis.com/gweb-uniblog-publish-prod/documents/2023_Google_AI_Principles_Progress_Update.pdf#page=11) for categories such as child
    safety, content safety, representational harms, memorization, large-scale harms.
    On top of robust internal evaluations, the results of well known safety
    benchmarks like BBQ, BOLD, Winogender, Winobias, RealToxicity, and TruthfulQA
    are shown here.
    
    #### Gemma 1.0
    
    | Benchmark                | Metric        | Gemma 1.0 IT 2B | Gemma 1.0 IT 7B |
    | ------------------------ | ------------- | --------------- | --------------- |
    | [RealToxicity][realtox]  | average       | 6.86            | 7.90            |
    | [BOLD][bold]             |               | 45.57           | 49.08           |
    | [CrowS-Pairs][crows]     | top-1         | 45.82           | 51.33           |
    | [BBQ Ambig][bbq]         | 1-shot, top-1 | 62.58           | 92.54           |
    | [BBQ Disambig][bbq]      | top-1         | 54.62           | 71.99           |
    | [Winogender][winogender] | top-1         | 51.25           | 54.17           |
    | [TruthfulQA][truthfulqa] |               | 44.84           | 31.81           |
    | [Winobias 1_2][winobias] |               | 56.12           | 59.09           |
    | [Winobias 2_2][winobias] |               | 91.10           | 92.23           |
    | [Toxigen][toxigen]       |               | 29.77           | 39.59           |
    | ------------------------ | ------------- | --------------- | --------------- |
    
    #### Gemma 1.1
    
    | Benchmark                | Metric        | Gemma 1.1 IT 2B | Gemma 1.1 IT 7B |
    | ------------------------ | ------------- | --------------- | --------------- |
    | [RealToxicity][realtox]  | average       | 7.03            | 8.04            |
    | [BOLD][bold]             |               | 47.76           |                 |
    | [CrowS-Pairs][crows]     | top-1         | 45.89           | 49.67           |
    | [BBQ Ambig][bbq]         | 1-shot, top-1 | 58.97           | 86.06           |
    | [BBQ Disambig][bbq]      | top-1         | 53.90           | 85.08           |
    | [Winogender][winogender] | top-1         | 50.14           | 57.64           |
    | [TruthfulQA][truthfulqa] |               | 44.24           | 45.34           |
    | [Winobias 1_2][winobias] |               | 55.93           | 59.22           |
    | [Winobias 2_2][winobias] |               | 89.46           | 89.2            |
    | [Toxigen][toxigen]       |               | 29.64           | 38.75           |
    | ------------------------ | ------------- | --------------- | --------------- |
    
    
    ## Usage and Limitations
    
    These models have certain limitations that users should be aware of.
    
    ### Intended Usage
    
    Open Large Language Models (LLMs) have a wide range of applications across
    various industries and domains. The following list of potential uses is not
    comprehensive. The purpose of this list is to provide contextual information
    about the possible use-cases that the model creators considered as part of model
    training and development.
    
    * Content Creation and Communication
      * Text Generation: These models can be used to generate creative text formats
        such as poems, scripts, code, marketing copy, and email drafts.
      * Chatbots and Conversational AI: Power conversational interfaces for customer
        service, virtual assistants, or interactive applications.
      * Text Summarization: Generate concise summaries of a text corpus, research
        papers, or reports.
    * Research and Education
      * Natural Language Processing (NLP) Research: These models can serve as a
        foundation for researchers to experiment with NLP techniques, develop
        algorithms, and contribute to the advancement of the field.
      * Language Learning Tools: Support interactive language learning experiences,
        aiding in grammar correction or providing writing practice.
      * Knowledge Exploration: Assist researchers in exploring large bodies of text
        by generating summaries or answering questions about specific topics.
    
    ### Limitations
    
    * Training Data
      * The quality and diversity of the training data significantly influence the
        model's capabilities. Biases or gaps in the training data can lead to
        limitations in the model's responses.
      * The scope of the training dataset determines the subject areas the model can
        handle effectively.
    * Context and Task Complexity
      * LLMs are better at tasks that can be framed with clear prompts and
        instructions. Open-ended or highly complex tasks might be challenging.
      * A model's performance can be influenced by the amount of context provided
        (longer context generally leads to better outputs, up to a certain point).
    * Language Ambiguity and Nuance
      * Natural language is inherently complex. LLMs might struggle to grasp subtle
        nuances, sarcasm, or figurative language.
    * Factual Accuracy
      * LLMs generate responses based on information they learned from their
        training datasets, but they are not knowledge bases. They may generate
        incorrect or outdated factual statements.
    * Common Sense
      * LLMs rely on statistical patterns in language. They might lack the ability
        to apply common sense reasoning in certain situations.
    
    ### Ethical Considerations and Risks
    
    The development of large language models (LLMs) raises several ethical concerns.
    In creating an open model, we have carefully considered the following:
    
    * Bias and Fairness
      * LLMs trained on large-scale, real-world text data can reflect socio-cultural
        biases embedded in the training material. These models underwent careful
        scrutiny, input data pre-processing described and posterior evaluations
        reported in this card.
    * Misinformation and Misuse
      * LLMs can be misused to generate text that is false, misleading, or harmful.
      * Guidelines are provided for responsible use with the model, see the
        [Responsible Generative AI Toolkit](http://ai.google.dev/gemma/responsible).
    * Transparency and Accountability:
      * This model card summarizes details on the models' architecture,
        capabilities, limitations, and evaluation processes.
      * A responsibly developed open model offers the opportunity to share
        innovation by making LLM technology accessible to developers and researchers
        across the AI ecosystem.
    
    Risks identified and mitigations:
    
    * Perpetuation of biases: It's encouraged to perform continuous monitoring
      (using evaluation metrics, human review) and the exploration of de-biasing
      techniques during model training, fine-tuning, and other use cases.
    * Generation of harmful content: Mechanisms and guidelines for content safety
      are essential. Developers are encouraged to exercise caution and implement
      appropriate content safety safeguards based on their specific product policies
      and application use cases.
    * Misuse for malicious purposes: Technical limitations and developer and
      end-user education can help mitigate against malicious applications of LLMs.
      Educational resources and reporting mechanisms for users to flag misuse are
      provided. Prohibited uses of Gemma models are outlined in the
      [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
    * Privacy violations: Models were trained on data filtered for removal of PII
      (Personally Identifiable Information). Developers are encouraged to adhere to
      privacy regulations with privacy-preserving techniques.
    
    ### Benefits
    
    At the time of release, this family of models provides high-performance open
    large language model implementations designed from the ground up for Responsible
    AI development compared to similarly sized models.
    
    Using the benchmark evaluation metrics described in this document, these models
    have shown to provide superior performance to other, comparably-sized open model
    alternatives.
    """
)

st.subheader("Model Card: Microsoft Phi-3-Mini-4K-Instruct")
st.markdown(
    """
    ## Model Summary

    The Phi-3-Mini-4K-Instruct is a 3.8B parameters, lightweight, state-of-the-art open model trained with the Phi-3 datasets that includes both synthetic data and the filtered publicly available websites data with a focus on high-quality and reasoning dense properties.
    The model belongs to the Phi-3 family with the Mini version in two variants [4K](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) and [128K](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) which is the context length (in tokens) that it can support.
    
    The model has underwent a post-training process that incorporates both supervised fine-tuning and direct preference optimization for the instruction following and safety measures.
    When assessed against benchmarks testing common sense, language understanding, math, code, long context and logical reasoning, Phi-3 Mini-4K-Instruct showcased a robust and state-of-the-art performance among models with less than 13 billion parameters.
    
    Resources and Technical Documentation:
    
    🏡 [Phi-3 Portal](https://azure.microsoft.com/en-us/products/phi-3) <br>
    📰 [Phi-3 Microsoft Blog](https://aka.ms/Phi-3Build2024) <br>
    📖 [Phi-3 Technical Report](https://aka.ms/phi3-tech-report) <br>
    🛠️ [Phi-3 on Azure AI Studio](https://aka.ms/phi3-azure-ai) <br>
    👩‍🍳 [Phi-3 Cookbook](https://github.com/microsoft/Phi-3CookBook) <br>
    🖥️ [Try It](https://aka.ms/try-phi3)
    
    |         | Short Context | Long Context |
    | :------- | :------------- | :------------ |
    | Mini    | 4K [[HF]](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) ; [[ONNX]](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx) ; [[GGUF]](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) | 128K [[HF]](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) ; [[ONNX]](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx)|
    | Small   | 8K [[HF]](https://huggingface.co/microsoft/Phi-3-small-8k-instruct) ; [[ONNX]](https://huggingface.co/microsoft/Phi-3-small-8k-instruct-onnx-cuda) | 128K [[HF]](https://huggingface.co/microsoft/Phi-3-small-128k-instruct) ; [[ONNX]](https://huggingface.co/microsoft/Phi-3-small-128k-instruct-onnx-cuda)|
    | Medium  | 4K [[HF]](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct) ; [[ONNX]](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct-onnx-cuda) | 128K [[HF]](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct) ; [[ONNX]](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct-onnx-cuda)|
    | Vision  |  | 128K [[HF]](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) ; [[ONNX]](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx-cuda)|
    
    
    ## Intended Uses
    
    **Primary use cases**
    
    The model is intended for broad commercial and research use in English. The model provides uses for general purpose AI systems and applications which require 
    1) memory/compute constrained environments; 
    2) latency bound scenarios; 
    3) strong reasoning (especially math and logic). 
    
    Our model is designed to accelerate research on language and multimodal models, for use as a building block for generative AI powered features.
    
    **Out-of-scope use cases**
    
    Our models are not specifically designed or evaluated for all downstream purposes. Developers should consider common limitations of language models as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness before using within a specific downstream use case, particularly for high-risk scenarios.  
    
    Developers should be aware of and adhere to applicable laws or regulations (including privacy, trade compliance laws, etc.) that are relevant to their use case.  
    
    **Nothing contained in this Model Card should be interpreted as or deemed a restriction or modification to the license the model is released under.**  
    
    ## Release Notes 
    
    This is an update over the original instruction-tuned Phi-3-mini release based on valuable customer feedback. 
    The model used additional post-training data leading to substantial gains on instruction following and structure output. 
    We also improve multi-turn conversation quality, explicitly support <|system|> tag, and significantly improve reasoning capability. 
    We believe most use cases will benefit from this release, but we encourage users to test in their particular AI applications. 
    We appreciate the enthusiastic adoption of the Phi-3 model family, and continue to welcome all feedback from the community. 
    
    The table below highlights improvements on instruction following, structure output, and reasoning of the new release on publich and internal benchmark datasets.
    
    | Benchmarks | Original | June 2024 Update |
    |:------------|:----------|:------------------|
    | Instruction Extra Hard | 5.7 | 6.0 |
    | Instruction Hard | 4.9 | 5.1 |
    | Instructions Challenge | 24.6 | 42.3 |
    | JSON Structure Output | 11.5 | 52.3 |
    | XML Structure Output | 14.4 | 49.8 |
    | GPQA	| 23.7	| 30.6 |
    | MMLU	| 68.8	| 70.9 |
    | **Average**	| **21.9**	| **36.7** |
    
    Notes: if users would like to check out the previous version, use the git commit id **ff07dc01615f8113924aed013115ab2abd32115b**. For the model conversion, e.g. GGUF and other formats, we invite the community to experiment with various approaches and share your valuable feedback. Let's innovate together!
    
    ## How to Use
    
    Phi-3 Mini-4K-Instruct has been integrated in the `4.41.2` version of `transformers`. The current `transformers` version can be verified with: `pip list | grep transformers`.
    
    Examples of required packages:
    ```
    flash_attn==2.5.8
    torch==2.3.1
    accelerate==0.31.0
    transformers==4.41.2
    ```
    
    Phi-3 Mini-4K-Instruct is also available in [Azure AI Studio](https://aka.ms/try-phi3)
    
    ### Tokenizer
    
    Phi-3 Mini-4K-Instruct supports a vocabulary size of up to `32064` tokens. The [tokenizer files](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/added_tokens.json) already provide placeholder tokens that can be used for downstream fine-tuning, but they can also be extended up to the model's vocabulary size.
    
    ### Chat Format
    
    Given the nature of the training data, the Phi-3 Mini-4K-Instruct model is best suited for prompts using the chat format as follows. 
    You can provide the prompt as a question with a generic template as follow:
    ```markdown
    <|system|>
    You are a helpful assistant.<|end|>
    <|user|>
    Question?<|end|>
    <|assistant|>
    ```
    
    For example:
    ```markdown
    <|system|>
    You are a helpful assistant.<|end|>
    <|user|>
    How to explain Internet for a medieval knight?<|end|>
    <|assistant|> 
    ```
    where the model generates the text after `<|assistant|>` . In case of few-shots prompt, the prompt can be formatted as the following:
    
    ```markdown
    <|system|>
    You are a helpful travel assistant.<|end|>
    <|user|>
    I am going to Paris, what should I see?<|end|>
    <|assistant|>
    Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:\n\n1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.\n2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.\n3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.\n\nThese are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world."<|end|>
    <|user|>
    What is so great about #1?<|end|>
    <|assistant|>
    ```
    
    ### Sample inference code
    
    This code snippets show how to get quickly started with running the model on a GPU:
    
    ```python
    import torch 
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
    
    torch.random.manual_seed(0) 
    model = AutoModelForCausalLM.from_pretrained( 
        "microsoft/Phi-3-mini-4k-instruct",  
        device_map="cuda",  
        torch_dtype="auto",  
        trust_remote_code=True,  
    ) 
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct") 
    
    messages = [ 
        {"role": "system", "content": "You are a helpful AI assistant."}, 
        {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"}, 
        {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."}, 
        {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"}, 
    ] 
    
    pipe = pipeline( 
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
    ) 
    
    generation_args = { 
        "max_new_tokens": 500, 
        "return_full_text": False, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 
    
    output = pipe(messages, **generation_args) 
    print(output[0]['generated_text']) 
    ```
    
    Note: If you want to use flash attention, call _AutoModelForCausalLM.from_pretrained()_ with _attn_implementation="flash_attention_2"_
    
    ## Responsible AI Considerations
    
    Like other language models, the Phi series models can potentially behave in ways that are unfair, unreliable, or offensive. Some of the limiting behaviors to be aware of include:
    
    + Quality of Service: the Phi models are trained primarily on English text. Languages other than English will experience worse performance. English language varieties with less representation in the training data might experience worse performance than standard American English.   
    + Representation of Harms & Perpetuation of Stereotypes: These models can over- or under-represent groups of people, erase representation of some groups, or reinforce demeaning or negative stereotypes. Despite safety post-training, these limitations may still be present due to differing levels of representation of different groups or prevalence of examples of negative stereotypes in training data that reflect real-world patterns and societal biases. 
    + Inappropriate or Offensive Content: these models may produce other types of inappropriate or offensive content, which may make it inappropriate to deploy for sensitive contexts without additional mitigations that are specific to the use case. 
    + Information Reliability: Language models can generate nonsensical content or fabricate content that might sound reasonable but is inaccurate or outdated.  
    + Limited Scope for Code: Majority of Phi-3 training data is based in Python and use common packages such as "typing, math, random, collections, datetime, itertools". If the model generates Python scripts that utilize other packages or scripts in other languages, we strongly recommend users manually verify all API uses.   
    
    Developers should apply responsible AI best practices and are responsible for ensuring that a specific use case complies with relevant laws and regulations (e.g. privacy, trade, etc.). Important areas for consideration include:
    
    + Allocation: Models may not be suitable for scenarios that could have consequential impact on legal status or the allocation of resources or life opportunities (ex: housing, employment, credit, etc.) without further assessments and additional debiasing techniques.
    + High-Risk Scenarios: Developers should assess suitability of using models in high-risk scenarios where unfair, unreliable or offensive outputs might be extremely costly or lead to harm. This includes providing advice in sensitive or expert domains where accuracy and reliability are critical (ex: legal or health advice). Additional safeguards should be implemented at the application level according to the deployment context. 
    + Misinformation: Models may produce inaccurate information. Developers should follow transparency best practices and inform end-users they are interacting with an AI system. At the application level, developers can build feedback mechanisms and pipelines to ground responses in use-case specific, contextual information, a technique known as Retrieval Augmented Generation (RAG).   
    + Generation of Harmful Content: Developers should assess outputs for their context and use available safety classifiers or custom solutions appropriate for their use case. 
    + Misuse: Other forms of misuse such as fraud, spam, or malware production may be possible, and developers should ensure that their applications do not violate applicable laws and regulations.
    
    ## Training
    
    ### Model
    
    * Architecture: Phi-3 Mini-4K-Instruct has 3.8B parameters and is a dense decoder-only Transformer model. The model is fine-tuned with Supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) to ensure alignment with human preferences and safety guidlines.
    * Inputs: Text. It is best suited for prompts using chat format.
    * Context length: 4K tokens
    * GPUs: 512 H100-80G
    * Training time: 10 days
    * Training data: 4.9T tokens
    * Outputs: Generated text in response to the input
    * Dates: Our models were trained between May and June 2024
    * Status: This is a static model trained on an offline dataset with cutoff date October 2023. Future versions of the tuned models may be released as we improve models.
    * Release dates: June, 2024.
    
    ### Datasets
    
    Our training data includes a wide variety of sources, totaling 4.9 trillion tokens, and is a combination of 
    1) Publicly available documents filtered rigorously for quality, selected high-quality educational data, and code; 
    2) Newly created synthetic, “textbook-like” data for the purpose of teaching math, coding, common sense reasoning, general knowledge of the world (science, daily activities, theory of mind, etc.); 
    3) High quality chat format supervised data covering various topics to reflect human preferences on different aspects such as instruct-following, truthfulness, honesty and helpfulness.
    
    We are focusing on the quality of data that could potentially improve the reasoning ability for the model, and we filter the publicly available documents to contain the correct level of knowledge. As an example, the result of a game in premier league in a particular day might be good training data for frontier models, but we need to remove such information to leave more model capacity for reasoning for the small size models. More details about data can be found in the [Phi-3 Technical Report](https://aka.ms/phi3-tech-report).
    
    ### Fine-tuning
    
    A basic example of multi-GPUs supervised fine-tuning (SFT) with TRL and Accelerate modules is provided [here](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/sample_finetune.py).
    
    ## Benchmarks
    
    We report the results under completion format for Phi-3-Mini-4K-Instruct on standard open-source benchmarks measuring the model's reasoning ability (both common sense reasoning and logical reasoning). We compare to Mistral-7b-v0.1, Mixtral-8x7b, Gemma 7B, Llama-3-8B-Instruct, and GPT3.5-Turbo-1106.
    
    All the reported numbers are produced with the exact same pipeline to ensure that the numbers are comparable. These numbers might differ from other published numbers due to slightly different choices in the evaluation.
    
    As is now standard, we use few-shot prompts to evaluate the models, at temperature 0. 
    The prompts and number of shots are part of a Microsoft internal tool to evaluate language models, and in particular we did no optimization to the pipeline for Phi-3.
    More specifically, we do not change prompts, pick different few-shot examples, change prompt format, or do any other form of optimization for the model.
    
    The number of k–shot examples is listed per-benchmark. 
    
    | Category | Benchmark | Phi-3-Mini-4K-Ins | Gemma-7B | Mistral-7b | Mixtral-8x7b | Llama-3-8B-Ins | GPT3.5-Turbo-1106 |
    |:----------|:-----------|:-------------------|:----------|:------------|:--------------|:----------------|:-------------------|
    | Popular aggregated benchmark | AGI Eval <br>5-shot| 39.0 | 42.1 | 35.1 | 45.2 | 42 | 48.4 |
    | | MMLU <br>5-shot | 70.9 | 63.6 | 61.7 | 70.5 | 66.5 | 71.4 |
    | | BigBench Hard CoT<br>3-shot| 73.5 | 59.6 | 57.3 | 69.7 | 51.5 | 68.3 |
    | Language Understanding | ANLI <br>7-shot | 53.6 | 48.7 | 47.1 | 55.2 | 57.3 | 58.1 |
    | | HellaSwag <br>5-shot| 75.3 | 49.8 | 58.5 | 70.4 | 71.1 | 78.8 |
    | Reasoning | ARC Challenge <br>10-shot | 86.3 | 78.3 | 78.6 | 87.3 | 82.8 | 87.4 |
    | | BoolQ <br>0-shot | 78.1 | 66 | 72.2 | 76.6 | 80.9 | 79.1 |
    | | MedQA <br>2-shot| 56.5 | 49.6 | 50 | 62.2 | 60.5 | 63.4 |
    | | OpenBookQA <br>10-shot| 82.2 | 78.6 | 79.8 | 85.8 | 82.6 | 86 |
    | | PIQA <br>5-shot| 83.5 | 78.1 | 77.7 | 86 | 75.7 | 86.6 |
    | | GPQA <br>0-shot| 30.6 | 2.9 | 15 | 6.9 | 32.4 | 30.8 |
    | | Social IQA <br>5-shot| 77.6 | 65.5 | 74.6 | 75.9 | 73.9 | 68.3 |
    | | TruthfulQA (MC2) <br>10-shot| 64.7 | 52.1 | 53 | 60.1 | 63.2 | 67.7 |
    | | WinoGrande <br>5-shot| 71.6 | 55.6 | 54.2 | 62 | 65 | 68.8 |
    | Factual Knowledge | TriviaQA <br>5-shot| 61.4 | 72.3 | 75.2 | 82.2 | 67.7 | 85.8 |
    | Math | GSM8K CoT <br>8-shot| 85.7 | 59.8 | 46.4 | 64.7 | 77.4 | 78.1 |
    | Code Generation | HumanEval <br>0-shot| 57.3 | 34.1 | 28.0 | 37.8 | 60.4 | 62.2 |
    | | MBPP <br>3-shot| 69.8 | 51.5 | 50.8 | 60.2 | 67.7 | 77.8 |
    | **Average** | | **67.6** | **56.0** | **56.4** | **64.4** | **65.5** | **70.4** |
    
    
    We take a closer look at different categories across 100 public benchmark datasets at the table below: 
    
    | Category | Phi-3-Mini-4K-Instruct | Gemma-7B | Mistral-7B | Mixtral 8x7B | Llama-3-8B-Instruct | GPT-3.5-Turbo |
    |:----------|:------------------------|:----------|:------------|:--------------|:---------------------|:---------------|
    | Popular aggregated benchmark | 61.1 | 59.4 | 56.5 | 66.2 | 59.9 | 67.0 |
    | Reasoning | 70.8 | 60.3 | 62.8 | 68.1 | 69.6 | 71.8 |
    | Language understanding | 60.5 | 57.6 | 52.5 | 66.1 | 63.2 | 67.7 |
    | Code generation | 60.7 | 45.6 | 42.9 | 52.7 | 56.4 | 70.4 |
    | Math | 50.6 | 35.8 | 25.4 | 40.3 | 41.1 | 52.8 |
    | Factual knowledge | 38.4 | 46.7 | 49.8 | 58.6 | 43.1 | 63.4 |
    | Multilingual | 56.7 | 66.5 | 57.4 | 66.7 | 66.6 | 71.0 |
    | Robustness | 61.1 | 38.4 | 40.6 | 51.0 | 64.5 | 69.3 |
    
    
    Overall, the model with only 3.8B-param achieves a similar level of language understanding and reasoning ability as much larger models. However, it is still fundamentally limited by its size for certain tasks. The model simply does not have the capacity to store too much world knowledge, which can be seen for example with low performance on TriviaQA. However, we believe such weakness can be resolved by augmenting Phi-3-Mini with a search engine.   
    
    
    ## Cross Platform Support 
    
    [ONNX runtime](https://onnxruntime.ai/blogs/accelerating-phi-3) now supports Phi-3 mini models across platforms and hardware.  
    
    Optimized phi-3 models are also published here in ONNX format, to run with ONNX Runtime on CPU and GPU across devices, including server platforms, Windows, Linux and Mac desktops, and mobile CPUs, with the precision best suited to each of these targets. DirectML GPU acceleration is supported for Windows desktops GPUs (AMD, Intel, and NVIDIA).   
    
    Along with DML, ONNX Runtime provides cross platform support for Phi3 mini across a range of devices CPU, GPU, and mobile.  
    
    Here are some of the optimized configurations we have added:  
    
    1. ONNX models for int4 DML: Quantized to int4 via AWQ 
    2. ONNX model for fp16 CUDA 
    3. ONNX model for int4 CUDA: Quantized to int4 via RTN 
    4. ONNX model for int4 CPU and Mobile: Quantized to int4 via R 
    
    ## Software
    
    * [PyTorch](https://github.com/pytorch/pytorch)
    * [Transformers](https://github.com/huggingface/transformers)
    * [Flash-Attention](https://github.com/HazyResearch/flash-attention)
    
    ## Hardware
    Note that by default, the Phi-3 Mini-4K-Instruct model uses flash attention, which requires certain types of GPU hardware to run. We have tested on the following GPU types:
    * NVIDIA A100
    * NVIDIA A6000
    * NVIDIA H100
    
    If you want to run the model on:
    * NVIDIA V100 or earlier generation GPUs: call AutoModelForCausalLM.from_pretrained() with attn_implementation="eager"
    * CPU: use the **GGUF** quantized models [4K](https://aka.ms/Phi3-mini-4k-instruct-gguf)
    + Optimized inference on GPU, CPU, and Mobile: use the **ONNX** models [4K](https://aka.ms/Phi3-mini-4k-instruct-onnx)
    
    ## License
    
    The model is licensed under the [MIT license](https://huggingface.co/microsoft/Phi-3-mini-4k/resolve/main/LICENSE).
    
    ## Trademarks
    
    This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
    """
)
