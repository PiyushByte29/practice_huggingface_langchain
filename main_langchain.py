# Import the Hugging Face pipeline utility for quick model loading and inference
from transformers import pipeline

# Import LangChain wrapper for Hugging Face pipelines
from langchain_huggingface import HuggingFacePipeline

# Import LangChain's PromptTemplate for dynamically formatting prompts
from langchain.prompts import PromptTemplate

# Import function to suppress warnings from Hugging Face Transformers
from transformers.utils.logging import set_verbosity_error

# Set Transformers logging level to "error" to avoid printing non-critical warnings
set_verbosity_error()

# ---------------------------------------------------------
# Create a summarization pipeline using Hugging Face
# "facebook/bart-large-cnn" → A pre-trained summarization model
# device=0 means using the first GPU (if available). Use -1 for CPU.
summarization_pipeline = pipeline(
    "summarization", model="facebook/bart-large-cnn", device=0)

# Wrap the HF pipeline in LangChain's HuggingFacePipeline to use it in a chain
summarizer = HuggingFacePipeline(pipeline=summarization_pipeline)

# ---------------------------------------------------------
# Create a refinement pipeline to further improve the summary
# Uses a slightly different model name ("facebook/bart-large") — note: this is a general BART model, not fine-tuned
refinement_pipeline = pipeline(
    "summarization", model="facebook/bart-large", device=0)

# Wrap the refinement model in LangChain's HuggingFacePipeline
refiner = HuggingFacePipeline(pipeline=refinement_pipeline)

# ---------------------------------------------------------
# Create a Question Answering (QA) pipeline
# "deepset/roberta-base-squad2" → A model fine-tuned for extractive QA tasks
qa_pipeline = pipeline("question-answering",
                       model="deepset/roberta-base-squad2", device=0)

# ---------------------------------------------------------
# Define a prompt template for summarization
# {length} → placeholder for desired summary length (short, medium, long)
# {text} → placeholder for the input text
summary_template = PromptTemplate.from_template(
    "Summarize the following text in a {length} way:\n\n{text}")

# Create a LangChain chain: prompt → summarizer → refiner
summarization_chain = summary_template | summarizer | refiner

# ---------------------------------------------------------
# Ask the user for the text they want summarized
text_to_summarize = input("\nEnter text to summarize:\n")

# Ask the user for desired summary length
length = input("\nEnter the length (short/medium/long): ")

# Pass user input into the summarization chain
# .invoke() sends a dictionary to the first item in the chain (summary_template).
# LangChain looks at the template and replaces:
# {text} → the value of "text" from the dictionary
# length} → the value of "length" from the dictionary

summary = summarization_chain.invoke(
    {"text": text_to_summarize, "length": length})

# Print the generated summary
print("\n🔹 **Generated Summary:**")
print(summary)

# ---------------------------------------------------------
# Interactive loop for asking questions about the summary
while True:
    # Ask user for a question
    question = input(
        "\nAsk a question about the summary (or type 'exit' to stop):\n")

    # Exit condition
    if question.lower() == "exit":
        break

    # Pass the question and summary as context into the QA pipeline
    qa_result = qa_pipeline(question=question, context=summary)

    # Display the answer extracted from the summary
    print("\n🔹 **Answer:**")
    print(qa_result["answer"])
