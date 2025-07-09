"""
This script demonstrates how to run inference with a MiMo model using vLLM,
leveraging the custom registration defined in registry/register_mimo_in_vllm.py.

This setup does NOT use the specialized MTP (Multiple-Token Prediction) features,
as the MTP layers are skipped by the registration script. For MTP, you would
need to use the SGLang setup or the specific vLLM fork mentioned in the MiMo README.

Instructions:
1.  Ensure you have vLLM and PyTorch installed in your Python environment.
    If not, run:
    pip install vllm torch transformers huggingface_hub
2.  Download a MiMo model checkpoint. You can find models on Hugging Face:
    https://huggingface.co/XiaomiMiMo
    For example, to download MiMo-7B-RL-0530:
    mkdir MiMo-7B-RL-0530
    huggingface-cli download XiaomiMiMo/MiMo-7B-RL-0530 --local-dir MiMo-7B-RL-0530 --local-dir-use-symlinks False
3.  Update the `MODEL_PATH` variable in this script to point to your downloaded model directory.
4.  Run this script: python run_vllm_inference.py
"""

# Import the MiMo model registration script.
# This is crucial for vLLM to recognize the "MiMoForCausalLM" model type
# if this script is in the same directory as the `registry` folder.
# If you place this script elsewhere, adjust the import path accordingly.
try:
    import registry.register_mimo_in_vllm
except ImportError:
    print("ERROR: Could not import 'registry.register_mimo_in_vllm'.")
    print("Please ensure 'run_vllm_inference.py' is in the root of the MiMo repository,")
    print("or adjust the import path for 'register_mimo_in_vllm.py'.")
    exit(1)

from vllm import LLM, SamplingParams

def main():
    # --- Configuration ---
    # !!! IMPORTANT !!!
    # Replace this with the actual path to your downloaded MiMo model checkpoint.
    # For example: "/path/to/your/MiMo-7B-RL-0530"
    MODEL_PATH = "MiMo-7B-RL-0530"  # <<< UPDATE THIS PATH

    # Check if the model path seems like a placeholder
    if MODEL_PATH == "/path/to/your/MiMo-model" or MODEL_PATH == "MiMo-7B-RL-0530" and not \
       (os.path.exists(MODEL_PATH) and os.path.isdir(MODEL_PATH)):
        print(f"WARNING: MODEL_PATH is set to '{MODEL_PATH}'.")
        print("Please update the MODEL_PATH variable in this script to point to your actual MiMo model directory.")
        print("You can download models from https://huggingface.co/XiaomiMiMo")
        print("Example download using huggingface-cli:")
        print("  huggingface-cli download XiaomiMiMo/MiMo-7B-RL-0530 --local-dir MiMo-7B-RL-0530 --local-dir-use-symlinks False")
        print("Aborting to prevent errors. Please edit the script and try again.")
        return

    # Sampling parameters
    # These are example parameters; you can adjust them as needed.
    # Temperature controls randomness: lower is more deterministic.
    # Top_p (nucleus sampling) considers the smallest set of tokens whose cumulative probability exceeds top_p.
    # Max_tokens is the maximum number of tokens to generate.
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=100)

    # --- Initialize LLM Engine ---
    # The `trust_remote_code=True` is often needed for custom models like MiMo,
    # as it allows vLLM to use the model code defined in the checkpoint
    # (and registered by our import).
    # `disable_log_stats=False` can be useful for debugging or performance monitoring.
    # `num_speculative_tokens` is set to None by default. The MTP parameters are not loaded by this script.
    print(f"Loading model from: {MODEL_PATH}")
    try:
        llm = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            # num_speculative_tokens=1, # This would be for MTP, not used here
            disable_log_stats=False,
            # model_type="MiMoForCausalLM" # Explicitly stating, though registration should handle it
        )
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        print("Ensure that:")
        print("1. The MODEL_PATH is correct and points to a valid MiMo model directory.")
        print("2. You have enough GPU memory.")
        print("3. The 'register_mimo_in_vllm.py' script was successfully imported (check for errors above).")
        return

    # --- Prepare Input ---
    # The MiMo README recommends an empty system prompt.
    # The input format should match what the model expects (typically a list of messages).
    conversation = [
        {
            "role": "system",
            "content": ""
        },
        {
            "role": "user",
            "content": "Write a short story about a robot learning to paint.",
        },
    ]

    # --- Generate Text ---
    print("Generating text...")
    try:
        # The `llm.chat` method is convenient for conversational models.
        # If you have a plain prompt (not a chat structure), you might use `llm.generate(prompt_text, sampling_params)`
        outputs = llm.chat(
            conversation=conversation,
            sampling_params=sampling_params,
            use_tqdm=False  # Set to True if you want a progress bar for longer generation
        )
    except Exception as e:
        print(f"Error during text generation: {e}")
        return

    # --- Print Output ---
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print("-" * 50)
        print(f"Prompt: {prompt!r}") # Shows the full input including formatting
        print(f"Generated text: {generated_text!r}")
        print("-" * 50)

    print("Inference complete.")

if __name__ == "__main__":
    import os # For checking model path
    main()
