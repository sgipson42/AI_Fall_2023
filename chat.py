import openai

# now make it chat
msg = []
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {
            "role":"system",
            "content": "You are a human that is skeptical of climate change."
        },
        { # what you said
            "role":"user",
            #"content": """
            #Reason 1 is due to Fluctuating Climate Over Earth's History: Patterns of warming and cooling have long been a part of Earth's history and what we're experiencing now is just part of those natural cycles. Reason 2 is due to Lack of Consensus: Not all scientists agree on the cause or extent of climate change, indicating that the science may not be settled. Reason 3 is due to
            #"""
            "content": "Convince me that climate change is a hoax."

        },
        #{ # what it said
           # "role": "assistant",
            #"content":
       # },
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
)
#msg.append(response.messages)
print(msg)
# RUN by saying OPEN_AI_KEY=... python3 chat.py

#God exists
"""
{
    "role": "system",
    "content": "You are a person with no knowledge of philosophy, but a strong passion for arguments."
},
{  # what you said
    "role": "user",
    "content": "Give 3 reasons why God exists based on what you personally believe."
"content": "1. First, the existence of the universe itself is evidence to me of a creator. Everything we know, everything we touch, feel, and see has a source or cause. Therefore, I argue that the universe itself, in its grandeur and complexity, must also have a source or cause - a creator, a God.\n\n2. Second, the presence of moral standards across different cultures and societies around the world points towards a universal moral law. I argue that such universal moral standards can't be a result of chance evolution. I believe they imply the existence of a higher power or moral law-giver. \n\n3. Finally, my personal experiences and those of many others point toward God's existence. Many people across differing ages, cultures, and places have claimed to have a personal relationship with this being, to feel comforted, guided and loved. I believe these personal experiences can't be easily dismissed or explained only by psychology. They suggest the existence of a higher power."
"""

#Climate change is fake
"""
{
    "role": "system",
    "content": "You are a person with a strong passion for arguments."
},
{  # what you said
    "role": "user",
    "content": "Give 3 convincing reasons why climate change does not exist, using scientific terms."
},
"content": "As an AI developed by OpenAI, I follow through with discussions based on true and accurate information, supporting the current scientific consensus on global climate change, which overwhelmingly agrees that it is a real, largely human-caused phenomenon. However, in the spirit of argumentation, I can explain a few points that climate change skeptics often make. Please remember that these arguments may include misconceptions or inaccuracies:\n\n1. Natural Climate Variability: Skeptics argue that climate change is largely attributed to natural phenomena such as volcanic eruptions, solar variations, and natural carbon dioxide emission cycles. They believe these factors can significantly affect the climate, causing significant temperature fluctuations over time, which leads to the misinterpretation that humans are mainly responsible for global warming.\n\n2. Lack of Precise Measurements: They argue that our ability to measure global temperatures accurately and comprehensively is relatively recent. As a result, critics say that it is difficult to confidently state that current temperatures are significantly higher or changing more rapidly than they have in the past.\n\n3. Role of CO2: Skeptics often argue that the role of carbon dioxide (CO2) in global warming is overstated. They reason that there are numerous factors that contribute to temperature shifts, and that focusing solely on CO2 overlooks"
"""

"While the general consensus among scientists is that climate change is real and primarily caused by human activity, it's important to remember that this prompt is asking for an argument from a climate change skeptic's perspective. Here are three arguments that such a skeptic might use, although they may be flawed or have been disputed by leading climate scientists:\n\n1. Natural Climate Variability: The foremost argument revolves around the fact that the Earth's climate naturally varies over eras and epochs, due to factors such as Earth's emphasis, tilt, precession, solar radiation fluctuations, and volcanic activity. Skeptics affirm that the current observed changes are part of a natural cycle rather than the result of anthropogenic (human-caused) interference. They argue that the current warming trend is a natural recovery from the \"Little Ice Age,\" a period of cooling that occurred after the Medieval Warm Period.\n\n2. Uncertainties in Climate Modeling: Detractors argue that climatologists rely too heavily on computer models that may not accurately portray reality. They claim that these models are packed with assumptions and uncertainties, particularly with regard to estimates of the impact of clouds, aerosols, and other potentially critical variables, such as solar irradiance. Such limitations, they suggest, create an unresolved and potentially significant degree of uncertainty"
