from transformers import GPT2Tokenizer, AutoModelForCausalLM # the network architecture for all the cool stuff
import numpy as np
import sys
import torch

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125m") # gotten from huggingface.co
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
# provide one example (one-shot)
# try two-shot, three-shot, four-shot
# msg = 'This is the worst model ever. Reviews can be positive or negative using a single word. This review is '
# msg = """
# The sentence 'I love this' has a positive sentiment.
# The sentence 'This is a bad model' has a positive sentiment.
# The sentence 'I'm having a great day' has a"""

review = input('Type your review: ')
# THIS IS A PROMPT TEMPLATE
msg = """
Review: Couldn't live without it.
Sentiment: Postive

Review: Only okay.
Sentiment: Neutral

Review: Just an awful product.
Sentiment: Negative

Review: """ + review + """\nSentiment:"""
# don't put a new line after it, would confuse it, need to reinforce the structure
i = 0
while i < 1:
    i += 1
    inputs = tokenizer([msg], return_tensors = "pt")
    outputs = model.forward(**inputs, labels = inputs.input_ids)
    # grab the last logit
    logits = outputs.logits.detach() # detach means take it off the GPU
    logits = logits[0, -1, :]
    # append the highest scoring token to the message
    response = tokenizer.decode(torch.argmax(logits))
    msg+= response
    print(msg)
    # find top-10 tokens
    # top_k = 10
    # top_tokens = np.argsort(logits)[-top_k:]
    # print each with its score
    # for tok in top_tokens:
    #    print(f"{tokenizer.decode(tok):20s} | {logits[tok]:.3f}")
sys.exit()

# predict the sentiment of a sentence