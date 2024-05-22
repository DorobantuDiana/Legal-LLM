from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

# Load the fine-tuned model and tokenizer
model_name = "fine-tuned-saullm-7b-romanian-legal"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def generate_response(prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response



# Simple chat interface

# Define the Gradio interface
def chat_with_model(user_input):
    response = generate_response(user_input)
    return response

iface = gr.Interface(
    fn=chat_with_model,
    inputs="text",
    outputs="text",
    title="Romanian Legal Chatbot",
    description="Ask questions about Romanian legal texts like the Constitution, Civil Code, or Penal Code."
)

# Launch the interface
iface.launch()
