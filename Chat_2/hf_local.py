# from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
# import os
# from dotenv import load_dotenv
#
# load_dotenv()
# token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# os.environ['HF_HOME'] = 'D:/huggingface_cache'
#
# llm = HuggingFacePipeline.from_model_id(
#     model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
#     task='text-generation',
#     pipeline_kwargs=dict(
#         temperature=0.5,
#         max_new_tokens=100
#     )
# )
# model = ChatHuggingFace(llm=llm)
#
# result = model.invoke("What is the capital of India")
#
# print(result.content)
# # import torch
# # print("CUDA available:", torch.cuda.is_available())
# # print("Device name:", torch.cuda.get_device_name(0))
# # print("Torch CUDA version:", torch.version.cuda)
# #
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
