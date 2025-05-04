import pandas as pd
from datasets import load_dataset

ds = load_dataset("lars1234/story_writing_benchmark", "average", split="train")
df = pd.DataFrame(ds)

df = df[df["language"] == "en"]

simple_question_labels = {
    "q1": "Grammar Checking",
    "q2": "Clarity",
    "q3": "Logic of Events",
    "q4": "Scene Purpose",
    "q5": "Internal Consistency",
    "q6": "Character Consistency",
    "q7": "Character Motivation",
    "q8": "Sentence Variety",
    "q9": "Cliché Avoidance",
    "q10": "Dialogue Naturalness",
    "q11": "Narrative Unpredictability",
    "q12": "Character Depth",
    "q13": "Realistic Interaction",
    "q14": "Engagement",
    "q15": "Plot Resolution"
}

detailed_question_descriptions = {
    "q1": "Grammar, spelling, and punctuation quality",
    "q2": "Clarity and understandability",
    "q3": "Logical connection between events and ideas",
    "q4": "Scene construction and purpose",
    "q5": "Internal consistency within the story's context",
    "q6": "Character consistency",
    "q7": "Character motivation and actions making sense",
    "q8": "Sentence pattern variety",
    "q9": "Avoidance of clichés and overused phrases",
    "q10": "Natural dialogue",
    "q11": "Avoidance of predictable narrative tropes",
    "q12": "Character depth and dimensionality",
    "q13": "Realistic character interactions",
    "q14": "Ability to hold reader interest",
    "q15": "Satisfying plot resolution"
}

long_df = df.melt(
    id_vars=["prompt", "story_text"],
    value_vars=list(simple_question_labels.keys()),
    var_name="q_key",
    value_name="human_score"
)

long_df["question"] = long_df["q_key"].map(detailed_question_descriptions).apply(
    lambda x: f"Evaluate the story's {x}"
)

long_df["question_type"] = long_df["q_key"].map(simple_question_labels)
long_df["golden_answer"] = None
long_df["answer_type"] = "story"
long_df["attempted_answer"] = long_df["story_text"]

final_df = long_df[[
    "question", "question_type", "golden_answer", "attempted_answer", "answer_type", "human_score"
]]

output_path = "../converted_dataset/story_benchmark_converted.csv"
final_df.to_csv(output_path, index=False)