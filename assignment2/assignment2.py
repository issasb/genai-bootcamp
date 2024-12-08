import os
from openai import AzureOpenAI
from PyPDF2 import PdfReader

# Retrieve the endpoint and key from the portal and set it as environment variables.
endpoint = os.environ.get('ENDPOINT')
key = os.environ.get('KEY')
model_name = "gpt-4"


client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_version="2024-02-01",
    api_key=key
)

def read_pdf(file_path):
  """
  Reads the text content from a PDF file.

  Args:
      file_path (str): Path to the PDF file.

  Returns:
      str: Extracted text content from the PDF.
  """
  try:
    with open(file_path, 'rb') as file:
      reader = PdfReader(file)
      content = ""
      for page in reader.pages:
        content += page.extract_text()
      return content
  except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    return ""

# Load the reference document from a PDF
reference_text = read_pdf('constitution_2.pdf')

def ask_and_complete(client, model_name):
  """
  Continuously asks the user for questions and uses the OpenAI API to complete them.
  Stops when the user types "quit".
  """
  while True:
    user_input = input("Ask me anything about the US constitution (or 'quit' to exit): ")
    if user_input.lower() == "quit":
      break
    
    completion = client.chat.completions.create(
      model=model_name,
      messages=[
          {
              "role": "user",
              "content": user_input + reference_text
          },
      ],
    )
    print(completion.choices[0].message.content)

ask_and_complete(client, model_name)
