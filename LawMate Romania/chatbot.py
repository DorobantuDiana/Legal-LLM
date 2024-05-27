from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

# Load the fine-tuned model and tokenizer
# model_name = "fine-tuned-saullm-7b-romanian-legal"
# model_name = "Equall/Saul-Instruct-v1"
model_name = "./fine-tuned-saullm-7b-romanian-legal"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def generate_response(prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Simple chat interface

# Define the Gradio interface
def chat_with_model(user_input):
    response = generate_response(user_input)
    return response


# URL of a background image related to the legal domain
background_image_url = 'url("https://t3.ftcdn.net/jpg/05/75/22/58/360_F_575225818_PQ2ZPHFw51yCcmieutB5bT843nPAPzo3.jpg")'

# Set up the Gradio interface with custom styling
css = """
body {
    background-image: """ + background_image_url + """;
    background-size: cover;
    font-family: Arial, sans-serif;
    color: #333333;
}
h1 {
    text-align: center;
    font-size: 2em;
    margin-top: 20px;
    color: #2C3E50;
}
p {
    text-align: center;
    font-size: 1.2em;
    color: #34495E;
}
#component-0 input[type="text"] {
    font-size: 1.2em;
}
#component-0 button {
    font-size: 1.2em;
    background-color: #2C3E50;
    color: #FFFFFF;
}
"""

iface = gr.Interface(
    fn=chat_with_model,
    inputs="text",
    outputs="text",
    title="LawMate Romania",
    description="Ask questions about Romanian legal texts (eg. Constitution, Law of Education, Civil Code etc.)",
    theme="gradio/glass"
)

# Launch the interface
iface.launch(
    server_name="0.0.0.0")