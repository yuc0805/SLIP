import yaml
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from string import Template

def build_prompts(texts):
    prompt_path = "/scratch/leo/workspace/HealthSensorSLM-Bench/prompt_bank/text_generate.md"
    with open(prompt_path, "r") as f:
        template = Template(f.read().strip())

    prompts = []
    for text in texts:
        # prompt = (
        #     "You are given one or multiple descriptions of the time series:\n"
        #     f"{text}\n\n"
        #     "Generate exactly 3 paraphrases. Requirements:\n"
        #     "1. The opening sentence that introduces the input sensor readings must also be rephrased in each paraphrase, "
        #     "but it must still mention every metric name from the original text.\n"
        #     "2. The rest of the description (such as trends, comparisons, interpretations) must also be fully preserved and rephrased.\n"
        #     "3. Do not omit or shorten any information.\n"
        #     "4. Do not add commentary.\n"
        #     "5. Write one paraphrase per line, plain text only, no numbering or bullets.\n\n"
        #     "Output format (strictly 3 lines):\n"
        #     "<Paraphrase 1>\n"
        #     "<Paraphrase 2>\n"
        #     "<Paraphrase 3>"
        # )
        prompt = template.substitute(text=text)
        prompts.append(prompt)
    return prompts


import re
def clean_llm_output(original_text, raw_output):
    """Split and clean LLM-generated text, prepend the original."""
    # aug_candidates = raw_output.replace("\r", "").strip().split("\n")
    # aug_candidates = [a.strip() for a in aug_candidates if a.strip()]
    # aug_candidates = [a for a in aug_candidates if a != original_text]
    parts = re.split(r'(?:Paraphrase\s*\d+:|\n{2,})', raw_output)
    parts = [p.strip() for p in parts if p.strip()]

    return [original_text] + parts

def main(input_json_path, output_path, llm_path):
    # Load dataset
    print('reading jsonl...')
    df = pd.read_json(input_json_path, lines=True, dtype_backend="pyarrow")
    print(f"Original dataset size: {df.shape}")


    all_texts = df["caption"].tolist()
    prompts = build_prompts(all_texts)

    # Generate augmentations
    from llm_utils import LLMClient  # import here to avoid global dependency

    llm_client = LLMClient(model_path=llm_path, engine="vllm",gpu_range=[2,3,4,5,6,7])
    llm_answers = llm_client.llm_batch_generate(prompts, use_chat_template=True)
    llm_client.kill()

    # Process outputs
    result_text = []
    for original_text, aug_text in tqdm(zip(all_texts, llm_answers)):
        result_text.append(clean_llm_output(original_text, aug_text))

    # Save augmented texts as JSONL
    text_path = os.path.join(output_path, "augmented_texts.jsonl")

    with open(text_path, "w") as f:
        for i, aug_text in enumerate(llm_answers):
            # convert the original row to dict
            row = df.iloc[i].to_dict()
            # replace 'caption' with a list containing the augmented and original text
            row["caption"] = [df.iloc[i]["caption"], aug_text]

            # save back to JSONL
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Augmented dataset saved to {text_path}, entries={len(df)}")

    # cleaning it:





if __name__ == '__main__':
    INPUT_JSON_PATH = 'data/pretrain_data.jsonl'
    OUTPUT_DIR = 'data/pretrain_data'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    LOCAL_LLM_PATH = yaml.safe_load(open("config/datagen_config.yaml"))["local_llm_path"]
    main(INPUT_JSON_PATH, OUTPUT_DIR, LOCAL_LLM_PATH)

    # python3 text_aug.py