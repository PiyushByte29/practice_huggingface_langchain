import torch  # Add this line to import the PyTorch library
from transformers import pipeline


print(torch.cuda.is_available)
print(torch.cuda.get_device_name)

model = pipeline("summarization", model="facebook/bart-large-cnn", device="0")
response = model("test to summarize")
print(response)
