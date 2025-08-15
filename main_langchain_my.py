from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers.utils.logging import set_verbosity_error

# Set Transformers logging level to "error" to avoid printing non-critical warnings
set_verbosity_error()

# Create a summarization pipeline using Hugging Face
# Create a refine summarization pipeline using Hugging Face
# ask Q & A

summarization_pipeline = pipeline(
    task="summarization", model="facebook/bart-large-cnn", device=0)
summarizer = HuggingFacePipeline(pipeline=summarization_pipeline)

refinement_pipeline = pipeline(
    task="summarization", model="facebook/bart-large", device=0)
refiner = HuggingFacePipeline(pipeline=refinement_pipeline)

qa_pipeline = pipeline(task="question-answering",
                       model="deepset/roberta-base-squad2", device=0)

summary_template = PromptTemplate.from_template(
    "Summarize the following text in a {length} way:\n\n{text}")

summarization_chain = summary_template | summarizer | refiner


text_to_summarize = input("\nEnter text to summarize:\n")
length = input("\nEnter the length (short/medium/long): ")

text_to_summarize = summarization_chain.invoke(
    {"text": text_to_summarize, "length": length})

# Print the generated summary
print("\n🔹 **Generated Summary:**")
print(text_to_summarize)

while True:
    question = input(
        "\nAsk a question about the summary (or type 'exit' to stop):\n")

    if (question.lower() == exit):
        break

    qa_result = qa_pipeline(question=question, context=text_to_summarize)
    # Display the answer extracted from the summary
    print("\n🔹 **Answer:**")
    print(qa_result["answer"])
