<div style="font-family: Arial, sans-serif; color: #333; background-color: #f8f8f8; padding: 20px; border-radius: 10px;">
<h1 style="color: #007bff;">LawMate Romania</h1>

## Overview

<p>LawMate Romania is a project focused on creating a Large Language Model (LLM) specialized in the Romanian legal domain. This model is designed to assist with various legal tasks by understanding and generating text based on Romanian legal documents. The project uses the <b>Equall/Saul-7B-Instruct-v1</b> pre-trained model from Hugging Face's library, specifically fine-tuned on Romanian legal texts like the Constitution and the Education Law.</p>

## Repo Structure

`documents/`: Contains text documents used for training the model, including the Romanian Constitution and the Education Law.

`training_ds/`: Contains the dataset files generated from the text documents for training purposes.

`env_llm.txt`: Lists the dependencies and environment settings required to run the project.

`main.py`: The main script for training and evaluating the Large Language Model (LLM).

`.gitignore`: Specifies files and directories to be ignored by Git to keep the repository clean.

`LawMate Romania/`: Includes the chatbot script and screenshots demonstrating example interactions.

## Steps to Replicate Results

1. **Set Up the Environment**:

   - Ensure you have Python installed.
   - Install the necessary dependencies by running:
     ```sh
     pip install -r env_llm.txt
     ```

2. **Prepare the PDF Files**:

   - Place your PDF files in the `documents/` directory.
   - The script will automatically extract and preprocess the text from these files.

3. **Fine-Tune the Model**:

   - Run the main training script `main.py` to fine-tune the pre-trained LLM on the provided dataset.

4. **Evaluate and Save the Model**:
   - After training, the script will evaluate the model's performance on the validation dataset.
   - The fine-tuned model will be saved for future use.
   </div>
