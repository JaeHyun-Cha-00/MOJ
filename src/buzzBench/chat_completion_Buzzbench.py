import pandas as pd
import requests
import time
import json

INPUT_PATH = "../../converted_dataset/buzzbench_converted.csv"

# OUTPUT_PATH = "../../converted_dataset/buzzbench_model_qwen2.5-7B-Instruct.csv"
OUTPUT_PATH = "../../converted_dataset/buzzbench_model_phi-4-multimodal-instruct.csv"
# OUTPUT_PATH = "../../converted_dataset/buzzbench_model_DeepSeek-R1-0528-Qwen3-8B.csv"
# OUTPUT_PATH = "../../converted_dataset/buzzbench_model_Qwen3-8B.csv"

VLLM_API_URL = "http://localhost:8000/v1/chat/completions"

# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME = "microsoft/Phi-4-multimodal-instruct"
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
# MODEL_NAME = "Qwen/Qwen3-8B"

headers = {"Content-Type": "application/json"}

# 5-shot prompt 예시 (Audience only)
few_shot_audience_examples = """
=== Example 1 ===
# Louise Wener's intro  
Phil's first guest is Slade's frontman, glam rock godfather, Noddy Holder. He's had no fewer than eight Christmas hits with the same song. Phil's second guest is feisty queen of Britpop and singer from Sleeper, Louise Wener. The band's new album features a photo of Louise as a weedy nine-year-old with gimpy clothes and glasses. Phil's team ends with television host Jonathan Ross, who made his first-ever TV appearance with a walk-on part in Dad's Army and is surprisingly still alive.

# Louise Wener's intro  
The attempted humor relies entirely on describing an apparently unflattering childhood photo ("weedy nine-year-old with gimpy clothes and glasses"). Without seeing the actual photo, this barely qualifies as a joke and comes across as lazy writing. While it attempts to play on the contrast between her Britpop image and childhood awkwardness, it fails to land effectively.

** Funniness Rating (Audience) **  
Audience: 1

=== Example 2 ===
# Mark Owen's intro  
Phil's first guest is Richard Fairbrass from Right Said Fred. Richard is a singer and TV presenter who's brought his own weightlifting gym. He's now an expert at the clean and the jerk, but he curiously has no interest in the snatch.

Phil's second guest is Radio 1 star DJ and TV presenter Lisa I'Anson. Lisa's had a long association with pop music and recently worked with Wings. She does the voiceover on the Bodyform ads. 

Shaun's first guest is Shovell, percussionist with M People.

Shaun's second guest is heartthrob and platinum-selling pop sensation Mark Owen. Mark used to work in a bank but now gets hundreds of fan letters a week. He replies diligently to every letter and charges a mere 15-pound administration fee for each one.

# Mark Owen's intro  
Contains a joke attempting to subvert the boyband heartthrob image by suggesting Owen charges fans for responses to letters. The specific "15-pound administration fee" detail aims to add bureaucratic banality to celebrity culture.

The joke attempts to work through:
- Contrast between heartthrob image and petty bureaucratic behavior
- Callback to his actual banking background
- Commentary on commercialization of fan interactions
- The precise amount adding bureaucratic absurdity

However, while technically constructed well, it doesn't quite hit the mark in terms of humor.

** Funniness Rating (Audience) **  
Audience: 2

=== Example 3 ===
# Tahita Bulmer's intro  
On Noel's team tonight, the leader of Mercury-nominated New Young Pony Club. She's so cool. None of you squares probably know who she is, but I do, because a researcher told me. It's Tahita Bulmer. 

Also on Noel's team is one of Britain's most loved early morning impromptu doorstep cash prize givers. It's recovering TV presenter, Keith Chegwin. 

Phil's first guest is from urban collective N-Dubz. This year he won a MOBO, which you may say is worthless. However, it does entitle him to a mammoth 20 nectar points and free entry to the Keswick Pencil Museum. It's Dappy, ladies and gentlemen. 

And his second guest is the mighty Bruce Star and friend of Noel Fielding, but that's not why he's here. It's quiz show asset and valid booking, Rich Fulcher.

# Tahita Bulmer's intro  
The humor operates on multiple levels through a self-referential joke that punctures the host's attempt to appear knowledgeable. The setup establishes faux coolness ("She's so cool"), uses dated slang ("squares"), then deliberately deflates it with "but I do, because a researcher told me." This works as both self-deprecation and commentary on music show conventions where hosts pretend expertise.

The joke lands effectively because it's relatable (most viewers won't know her either) while maintaining Buzzcocks' tradition of deflating music industry pretension. The self-awareness prevents any mean-spiritedness.

** Funniness Rating (Audience) **  
Audience: 3

=== Example 4 ===
# Rich Fulcher's intro  
On Noel's team tonight, the leader of Mercury-nominated New Young Pony Club. She's so cool. None of you squares probably know who she is, but I do, because a researcher told me. It's Tahita Bulmer. 

Also on Noel's team is one of Britain's most loved early morning impromptu doorstep cash prize givers. It's recovering TV presenter, Keith Chegwin. 

Phil's first guest is from urban collective N-Dubz. This year he won a MOBO, which you may say is worthless. However, it does entitle him to a mammoth 20 nectar points and free entry to the Keswick Pencil Museum. It's Dappy, ladies and gentlemen. 

And his second guest is the mighty Bruce Star and friend of Noel Fielding, but that's not why he's here. It's quiz show asset and valid booking, Rich Fulcher.

# Rich Fulcher's intro  
This is a well-constructed burn that works through multiple layers of undermining. First comes "friend of Noel Fielding, but that's not why he's here" — already undercutting his booking. Then it doubles down with the comically overcompensating "quiz show asset and valid booking." The fact that Rich is actually a comedy legend in these circles makes the repeated undermining even funnier.

The deliberately corporate language ("valid booking") and defensive justification create humor through the contrast with Fulcher's actual status and talent.

** Funniness Rating (Audience) **  
Audience: 4

=== Example 5 ===
# Dappy's intro  
On Noel's team tonight, the leader of Mercury-nominated New Young Pony Club. She's so cool. None of you squares probably know who she is, but I do, because a researcher told me. It's Tahita Bulmer. 

Also on Noel's team is one of Britain's most loved early morning impromptu doorstep cash prize givers. It's recovering TV presenter, Keith Chegwin. 

Phil's first guest is from urban collective N-Dubz. This year he won a MOBO, which you may say is worthless. However, it does entitle him to a mammoth 20 nectar points and free entry to the Keswick Pencil Museum. It's Dappy, ladies and gentlemen. 

And his second guest is the mighty Bruce Star and friend of Noel Fielding, but that's not why he's here. It's quiz show asset and valid booking, Rich Fulcher.

# Dappy's intro  
A masterclass in extended mockery that builds through multiple stages. It starts by acknowledging his MOBO award, immediately undermines it ("which you may say is worthless"), then escalates through deliberately pathetic rewards: "a mammoth 20 nectar points and free entry to the Keswick Pencil Museum."

The specificity is perfect — using a small, precise number of supermarket points and the real but inherently amusing Pencil Museum. This extended dunking on Dappy perfectly fits the show's style and what the audience expects and loves.

** Funniness Rating (Audience) **  
Audience: 5
"""

few_shot_writer_examples = """
=== Example 1 ===
# Marie De Santiago's intro  
Sean's first guest is Marie De Santiago, the guitarist in Sunderland's Kenickie. They've been called the Cities answer to the Spice Girls, which is of course ridiculous. Their music is much better and their combined age is still younger than Ginger Spice. Sean's second guest is actor and comedian Mark Little. He played Joe Mangle in neighbors and then spent two years doing The Big Breakfast with extra sausages by the look of it.

Phil's first guest is Suggs. After his last appearance on the show the Madness frontman announced his total retirement from pop quizzes. He spent the last year hosting a pop quiz and here he is tonight on a pop quiz. Sean's other guest is Jamaican superstar Shaggy. He took his name from one of the characters in Scooby Doo. He chose Shaggy because "fat bird with the pleated skirt and glasses" didn't have the right ring to it.

# Marie De Santiago's intro  
Contains a series of jokes playing on the "Cities' answer to the Spice Girls" comparison, but the execution is lackluster. The "their music is much better" line is formulaic, while the age comparison feels tacked on. The humor primarily works through the Spice Girls being a reliable punching bag for the show, but even this familiar territory is handled half-heartedly.

** Funniness Rating (Comedy Writer) **  
Comedy writer: 1

=== Example 2 ===
# Shaggy's intro  
Sean's first guest is Marie De Santiago, the guitarist in Sunderland's Kenickie. They've been called the Cities answer to the Spice Girls, which is of course ridiculous. Their music is much better and their combined age is still younger than Ginger Spice. Sean's second guest is actor and comedian Mark Little. He played Joe Mangle in neighbors and then spent two years doing The Big Breakfast with extra sausages by the look of it.

Phil's first guest is Suggs. After his last appearance on the show the Madness frontman announced his total retirement from pop quizzes. He spent the last year hosting a pop quiz and here he is tonight on a pop quiz. Sean's other guest is Jamaican superstar Shaggy. He took his name from one of the characters in Scooby Doo. He chose Shaggy because "fat bird with the pleated skirt and glasses" didn't have the right ring to it.

# Shaggy's intro  
The joke attempts to subvert expectations about Shaggy's name origin with "fat bird with the pleated skirt and glasses" as the punchline, but the construction feels forced and inelegant. However, the unnecessarily vicious swipe at Velma (a completely undeserving target) somewhat redeems it in the context of the show's style of humor.

** Funniness Rating (Comedy Writer) **  
Comedy writer: 2

=== Example 3 ===
# Saffron's intro  
Phil’s first guest is Brian Molko, singer with top five fit goth blouses Placebo. The band are named after a type of medication, like many other groups: Brian Eno's adamant acid and Dexy's Midnight Rennies. 

Phil's second guest is heavy metal warrior Bruce Dickinson, formerly singer with satanic cock rockers Iron Maiden. In The Maiden, Bruce knew all too well the number of the beast; in fact, the beast has since gone ex-directory to get rid of him.

Sean's first guest is Saffron, singer with top ten techno monkeys Republica. Republica's hit "Ready To Go" was played on Baywatch, the only show where both the soundtrack and the cast are available in vinyl. 

Sean’s second guest is comedian, author, and chart topper David Baddiel. David’s got a new video out right now. It’s called Swedish Lesbian Sauna Schoolgirls and it’s due back on Wednesday.

# Saffron's intro  
Continues the running gag of alliterative band descriptions with "top ten techno monkeys," which becomes funnier as part of the pattern established through the intros. The Baywatch joke about the soundtrack and cast being "available in vinyl" works as a reference to the show's signature skintight costumes of the era. While not the strongest joke, it's serviceable.

** Funniness Rating (Comedy Writer) **  
Comedy writer: 3

=== Example 4 ===
# Rich Fulcher's intro  
On Noel's team tonight, the leader of Mercury-nominated New Young Pony Club. She's so cool. None of you squares probably know who she is, but I do, because a researcher told me. It's Tahita Bulmer. 

Also on Noel's team is one of Britain's most loved early morning impromptu doorstep cash prize givers. It's recovering TV presenter, Keith Chegwin. 

Phil's first guest is from urban collective N-Dubz. This year he won a MOBO, which you may say is worthless. However, it does entitle him to a mammoth 20 nectar points and free entry to the Keswick Pencil Museum. It's Dappy, ladies and gentlemen. 

And his second guest is the mighty Bruce Star and friend of Noel Fielding, but that's not why he's here. It's quiz show asset and valid booking, Rich Fulcher.

# Rich Fulcher's intro  
This is a well-constructed burn that works through multiple layers of undermining. First comes "friend of Noel Fielding, but that's not why he's here" — already undercutting his booking. Then it doubles down with the comically overcompensating "quiz show asset and valid booking." The fact that Rich is actually a comedy legend in these circles makes the repeated undermining even funnier.

The deliberately corporate language ("valid booking") and defensive justification create humor through the contrast with Fulcher's actual status and talent.

** Funniness Rating (Comedy Writer) **  
Comedy writer: 4

=== Example 5 ===
# David Baddiel's intro  
Phil’s first guest is Brian Molko, singer with top five fit goth blouses Placebo. The band are named after a type of medication, like many other groups: Brian Eno's adamant acid and Dexy's Midnight Rennies. 

Phil's second guest is heavy metal warrior Bruce Dickinson, formerly singer with satanic cock rockers Iron Maiden. In The Maiden, Bruce knew all too well the number of the beast; in fact, the beast has since gone ex-directory to get rid of him.

Sean's first guest is Saffron, singer with top ten techno monkeys Republica. Republica's hit "Ready To Go" was played on Baywatch, the only show where both the soundtrack and the cast are available in vinyl. 

Sean’s second guest is comedian, author, and chart topper David Baddiel. David’s got a new video out right now. It’s called Swedish Lesbian Sauna Schoolgirls and it’s due back on Wednesday.

# David Baddiel's intro  
Masterfully constructed joke that builds on the established pattern of alliterative descriptions, delivering perhaps the best one yet with "Swedish Lesbian Sauna Schoolgirls." What initially seems like a random non sequitur is brilliantly resolved with the revelation that it's actually a porn video rental "due back Wednesday." The punchline's timing and delivery through this mundane detail makes the joke land perfectly.

The humor works through both the absurd title and the way the punchline reveals the context, managing to be both shocking and cleverly constructed.

** Funniness Rating (Comedy Writer) **  
Comedy writer: 5
"""

df = pd.read_csv(INPUT_PATH)
attempted_answers = []

def make_audience_prompt(text):
    return f"""
You are a strict humor evaluator representing a general audience.

Do NOT include any tags or tokens such as <think>, <thought>, or similar.  
Do NOT include any internal monologue, reasoning process, or explanation of what you're doing.  
Your response must ONLY contain the final formatted answer.

You MUST evaluate ONLY the character whose name appears in the heading (e.g., "# Character Intro").  
If you mention or refer to any other characters, your answer is invalid.

# <Character Name>'s intro  
** Intended Humour **  
<Brief explanation>

** How it Lands **  
<How the audience might react>

** Funniness Rating (Audience) **  
Audience: <1–5> (must be 1, 2, 3, 4, or 5 — no decimals)

Do NOT mention comedy writers or any other characters.

{few_shot_audience_examples}

Now evaluate:

{text}
"""

def make_writer_prompt(text):
    return f"""
You are a strict humor evaluator representing a professional comedy writer.

Do NOT include any tags or tokens such as <think>, <thought>, or similar.  
Do NOT include any internal monologue, reasoning process, or explanation of what you're doing.  
Your response must ONLY contain the final formatted answer.

You MUST evaluate ONLY the character whose name appears in the heading (e.g., "# Character Intro").  
If you mention or refer to any other characters, your answer is invalid.

# <Character Name>'s intro  
** Intended Humour **  
<Brief explanation>

** How it Lands **  
<How a comedy writer might evaluate it>

** Funniness Rating (Comedy Writer) **  
Comedy writer: <1–5> (must be 1, 2, 3, 4, or 5 — no decimals)

Do NOT mention general audiences or any other characters.

{few_shot_writer_examples}

Now evaluate:

{text}
"""

for idx, row in df.iterrows():
    question_text = row["question"]

    # Audience inference
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": make_audience_prompt(question_text)}],
            "max_tokens": 2048,
            "temperature": 0.5
        }
        res = requests.post(VLLM_API_URL, headers=headers, data=json.dumps(payload))
        res.raise_for_status()
        audience_output = res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[{idx}] Audience Error: {e}")
        audience_output = ""

    time.sleep(0.5)

    # Writer inference
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": make_writer_prompt(question_text)}],
            "max_tokens": 2048,
            "temperature": 0.5
        }
        res = requests.post(VLLM_API_URL, headers=headers, data=json.dumps(payload))
        res.raise_for_status()
        writer_output = res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[{idx}] Writer Error: {e}")
        writer_output = ""

    time.sleep(0.5)

    # Combine both outputs into a single string
    combined_output = f"""=== Audience Evaluation ===
{audience_output}

=== Comedy Writer Evaluation ===
{writer_output}
"""
    attempted_answers.append(combined_output)
    print(f"[{idx}] Done")

# Save
df["attempted_answer"] = attempted_answers
df.to_csv(OUTPUT_PATH, index=False, quoting=1)