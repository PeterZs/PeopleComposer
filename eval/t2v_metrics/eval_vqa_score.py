# conda activate t2v
import t2v_metrics
import json
from tqdm import tqdm

clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl') # our recommended scoring model

result_summary_path = "<YOUR_RESULTS_SUMMARY_JSON>"
with open(result_summary_path, "r") as f:
    result_summary = json.load(f)

max_people = None # None means all people

for prompt_idx, prompt_data in tqdm(result_summary.items()):
    num_people = prompt_data['number_people']
    if max_people is not None and num_people > max_people:
        continue
    caption = prompt_data['update2']['prompt']
    for folder_type in ['update2', 'update1', 'original']:
        if not prompt_data[folder_type]:
            continue
        for seed_idx, seed_data in prompt_data[folder_type]['results'].items():
            if 'score' in result_summary[prompt_idx][folder_type]['results'][seed_idx]: # no repeat calculation
                continue
            image_path = seed_data['generation_result']
            score = clip_flant5_score(images=[image_path], texts=[caption])
            # Convert tensor to float for JSON serialization
            if hasattr(score, 'item'):
                score_value = score.item()
            else:
                score_value = float(score)
            print(f"Prompt {prompt_idx}, {folder_type}, {seed_idx}, {score_value}")
            result_summary[prompt_idx][folder_type]['results'][seed_idx]['score'] = score_value


result_summary_w_score_path = "<YOUR_RESULTS_SUMMARY_WITH_SCORE_JSON>"
with open(result_summary_w_score_path, "w") as f:
    json.dump(result_summary, f)
