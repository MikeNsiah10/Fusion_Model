from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json, re

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Qwen/Qwen1.5-1.8B-Instruct"

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,   # preferred for GPU
    device_map="auto"
)

# Function to add turn labels
def add_turn_labels(conversation):
    prev_speaker = None
    for turn in conversation["turns"]:
        if prev_speaker is None:
            turn["label"] = 0  # first turn always 0
        else:
            turn["label"] = 0 if turn["speaker"] == prev_speaker else 1
        prev_speaker = turn["speaker"]
    return conversation

# Prompt (no labels in output)
prompt_template = """
You are a dataset generator that produces realistic multi-turn conversations between speakers A and B. 
Speakers should not always alternate turns. Sometimes a speaker continues multiple times before the 
other responds.

The output must be in JSON format with the following structure (without labels):

{
  "conversation_id": "conv_001",
  "turns": [
    {"speaker": "A", "text": "Hey, are you coming to the meeting later?"},
    {"speaker": "B", "text": "Not sure, what time is it?"},
    {"speaker": "B", "text": "Actually, I might join if it’s after lunch."},
    {"speaker": "A", "text": "Yeah, it’s at 2pm."}
  ]
}

Rules:
- Do NOT include labels, only speaker and text.
- Generate 5 conversations per request.
- Each conversation should have 8–12 turns.
- Topics: daily life, work, study, hobbies, travel.
- Output ONLY valid JSON, no extra text or markdown.
"""

# Generate in batches until ~14,000 conversations
target_convos = 14000
batch_size = 5
conversations = []
conv_id = 1

while len(conversations) < target_convos:
    inputs = tokenizer(prompt_template, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1000,
        temperature=0.9,
        do_sample=True
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract JSON block
    start = text.find("{")
    end = text.rfind("}") + 1
    json_text = text[start:end]

    # Cleanup common issues
    json_text = re.sub(r"^```.*|```$", "", json_text, flags=re.MULTILINE).strip()
    json_text = re.sub(r",\s*}", "}", json_text)
    json_text = re.sub(r",\s*]", "]", json_text)

    try:
        data = json.loads(json_text)
        if isinstance(data, dict):  # wrap single dict
            data = [data]

        # Add labels
        for conv in data:
            conv["conversation_id"] = f"conv_{conv_id:05d}"
            conv_id += 1
            conversations.append(add_turn_labels(conv))

    except Exception as e:
        print("JSON parse failed, skipping batch:", e)

    print(f"Generated {len(conversations)} / {target_convos} conversations")

# Save to JSONL
save_path = "/share/users/student/m/mnsiah/distil_student_logs/train.jsonl"
with open(save_path, "w", encoding="utf-8") as f:
    for conv in conversations:
        f.write(json.dumps(conv) + "\n")

print(f"Saved dataset with {len(conversations)} conversations to {save_path}")
