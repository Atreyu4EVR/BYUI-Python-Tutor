import streamlit as st


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
