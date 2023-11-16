# https://arxiv.org/pdf/2009.12240.pdf

"""
Data: prompt is a string; scheme is a list of lines where each line is a list of segments where
segmenti = hsj , rj , ej i for j = 1...n such that sj is the number of syllables in the segment, rj is
rhyme index or string, and ej is an optional signifier that this segment should end a sentence; and
recontextualize? is a boolean.
Result: A list of strings constituting the lines of the lyrics
context ← prompt
for line in lines do
if recontextualize? = true then
Insert prompt in context after last occurring period
for segment in line do
target_syllables ← number of syllables specified in segment
rhyme_index ← rhyme index specified in segment
end? ← true if segment specifies the segment ends in a period
if rhyme_index is a string or rhyme_index in rhyme_map then
end_targets ← pick rhyme words or use rhyme_index
for target in end_targets do
candidates ← candidates+generate_rhyme_lines (target, context,
target_syllables, end?)
else
if end? = true then
candidates ← candidates+generate_terminal_non_rhyme_lines (context,
target_syllables)
else
candidates ← candidates+generate_non_rhyme_lines (context, target_syllables)
best ←pick_best_candidate (candidates, context)
context = context + best
final_segments ← f inal_segments + best
final_lines = f inal_lines + f inal_segments
"""

from transformers import GPT2Tokenizer, GPT2LMHeadModel, XLNetTokenizer, XLNetLMHeadModel
import torch
from nltk.corpus import cmudict
from syllables import nsyl, remove_punctuation

cmu_save = cmudict.entries()
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
xlnet_model = XLNetLMHeadModel.from_pretrained("xlnet-base-cased")

# https://stackoverflow.com/questions/25714531/find-rhyme-using-nltk-in-python
def rhyme(inp, level):
    syllables = [(word, syl) for word, syl in cmu_save if word == inp]
    rhymes = []
    for (word, syllable) in syllables:
        rhymes += [word for word, pron in cmu_save if pron[-level:] == syllable[-level:]]
    return set(rhymes)

def check_rhyme(input, word, level=1):
    return word.lower() in rhyme(input, level)

prompt = "You write parody songs from existing song lyrics.\n\nFor example for a machine learning parody:\nSong Line: I knew it when I met him (ayy), I loved him when I left him\nParody Line: I found the cost in batches, I sold it when I left it\n\nWrite a song to help students learn calculus differentiation rules"

lines = [
    "Havana, ooh na-na",
    "Half of my heart is in Havana, ooh-na-na (ay, ay)",
    "He took me back to East Atlanta, na-na-na",
    "Oh, but my heart is in Havana (ay)",
    "There's somethin' 'bout his manners (uh huh)",
    "Havana, ooh na-na (uh)",
]

""" for i in range(10):
    r = rhyme("anonymous", i+1)
    if len(r) < 500:
        print(r)
    print(len(r)) """

context = ""
context += prompt
recontextualize = True
rhyme_map = []
final_lines = ""

# Implement the function to generate non-rhyme lines using GPT-2
def generate_non_rhyme_lines(context, target_syllables, tokinzer=gpt2_tokenizer, model=gpt2_model):
    # Prepare the input text with the context
    input_text = context + " "
    
    # Encode the input text
    input_ids = tokinzer.encode(input_text, return_tensors="pt")
    
    # Generate text using the GPT-2 model
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens= 5,  # Adjust the length as per your requirements
            num_return_sequences=1,
            pad_token_id=tokinzer.eos_token_id,
            do_sample=True,
            #top_k=50,
            temperature=0.7,
        )
    
    # Decode and return the generated non-rhyme line
    non_rhyme_line = tokinzer.decode(output[0], skip_special_tokens=True)
    print(non_rhyme_line)
    # Filter out lines with incorrect syllable count
    while nsyl(remove_punctuation(non_rhyme_line[len(input_text):])) != target_syllables:
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens= 5,  # Adjust the length as per your requirements
            num_return_sequences=1,
            pad_token_id=tokinzer.eos_token_id,
            do_sample=True,
            #top_k=50,
            temperature=0.7,
        )
        non_rhyme_line = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
        print(non_rhyme_line)

    return non_rhyme_line[len(input_text):]

def pick_best_candidate(candidates, context):
    print(candidates)
    return candidates[0]

for line in lines:
    if recontextualize:
        # insert prompt in context after last occurring period
        pass
    final_segments = ""
    context += f"\nSong Line: {line}\nParody Line: "
    
    target_syllables = nsyl(remove_punctuation(line))
    rhyme_index = None
    end = True if line[-1] == "." else False

    candidates = ""

    if rhyme_index is not None or rhyme_index in rhyme_map:
        end_targets = rhyme(line, 1)
        for target in end_targets:
            candidates += generate_rhyme_lines(target, context, target_syllables, end)

    else:
        if end:
            candidates += generate_terminal_non_rhyme_lines(context, target_syllables)
        else:
            candidates += generate_non_rhyme_lines(context, target_syllables, tokinzer=xlnet_tokenizer, model=xlnet_model)
    
    best = pick_best_candidate(candidates, context)
    context += best
    final_segments += best
    final_lines += final_segments + "/n"
    print("GOT A LINE")