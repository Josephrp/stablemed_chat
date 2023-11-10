from transformers import AutoTokenizer, MistralForCausalLM
import torch
import gradio as gr
import random
from textwrap import wrap
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, MistralForCausalLM
from peft import PeftModel, PeftConfig
import torch
import gradio as gr
import os

hf_token = os.environ.get('HUGGINGFACE_TOKEN')

# Functions to Wrap the Prompt Correctly
def wrap_text(text, width=90):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text
def multimodal_prompt(user_input, system_prompt="You are an expert medical analyst:"):

    # Combine user input and system prompt
    formatted_input = f"[INSTRUCTION]{system_prompt}[QUESTION]{user_input}"

    # Encode the input text
    encodeds = tokenizer(formatted_input, return_tensors="pt", add_special_tokens=False)
    model_inputs = encodeds.to(device)

    # Generate a response using the model
    output = model.generate(
        **model_inputs,
        max_length=max_length,
        use_cache=True,
        early_stopping=True,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.eos_token_id,
        temperature=0.1,
        do_sample=True
    )

    # Decode the response
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return response_text

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Use the base model's ID
base_model_id = "stabilityai/stablelm-3b-4e1t"
model_directory = "Tonic/stablemed"

# Instantiate the Tokenizer
tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-3b-4e1t", token=hf_token, trust_remote_code=True, padding_side="left")
# tokenizer = AutoTokenizer.from_pretrained("Tonic/stablemed", trust_remote_code=True, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# Load the PEFT model
peft_config = PeftConfig.from_pretrained("Tonic/stablemed", token=hf_token)
peft_model = MistralForCausalLM.from_pretrained("stabilityai/stablelm-3b-4e1t", token=hf_token, trust_remote_code=True)
peft_model = PeftModel.from_pretrained(peft_model, "Tonic/stablemed", token=hf_token)

class ChatBot:
    def __init__(self):
        self.history = []

    def predict(self, user_input, system_prompt="You are an expert medical analyst:"):
        # Combine user input and system prompt
        formatted_input = f"[INSTRUCTION:]{system_prompt}[QUESTION:] {user_input}"

        # Encode user input
        user_input_ids = tokenizer.encode(formatted_input, return_tensors="pt")

        # Concatenate the user input with chat history
        if len(self.history) > 0:
            chat_history_ids = torch.cat([self.history, user_input_ids], dim=-1)
        else:
            chat_history_ids = user_input_ids

        # Generate a response using the PEFT model
        response = peft_model.generate(input_ids=chat_history_ids, max_length=400, pad_token_id=tokenizer.eos_token_id)

        # Update chat history
        self.history = chat_history_ids

        # Decode and return the response
        response_text = tokenizer.decode(response[0], skip_special_tokens=True)
        return response_text

bot = ChatBot()

title = "👋🏻Welcome to Tonic's StableMed Chat🚀"
description = """
You can use this Space to test out the current model [StableMed](https://huggingface.co/Tonic/stablemed) or You can also use 😷StableMed⚕️ on your own data & in your own way by cloning this space. 🧬🔬🔍 Simply click here: <a style="display:inline-block" href="https://huggingface.co/spaces/Tonic/StableMed_Chat?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a></h3> 
# Join us : 🌟TeamTonic🌟 is always making cool demos! Join our active builder's🛠️community on 👻Discord: [Discord](https://discord.gg/GWpVpekp) On 🤗Huggingface: [TeamTonic](https://huggingface.co/TeamTonic) & [MultiTransformer](https://huggingface.co/MultiTransformer) On 🌐Github: [Polytonic](https://github.com/tonic-ai) & contribute to 🌟 [PolyGPT](https://github.com/tonic-ai/polygpt-alpha)
"""
examples = [["What is the proper treatment for buccal herpes?", "Please provide information on the most effective antiviral medications and home remedies for treating buccal herpes."]]

iface = gr.Interface(
    fn=bot.predict,
    title=title,
    description=description,
    examples=examples,
    inputs=["text", "text"],  # Take user input and system prompt separately
    outputs="text",
    theme="ParityError/Anime"
)

iface.launch()
