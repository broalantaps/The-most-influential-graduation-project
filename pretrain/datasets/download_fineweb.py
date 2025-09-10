from huggingface_hub import snapshot_download
folder = snapshot_download(
    "HuggingFaceFW/fineweb", 
    repo_type="dataset",
    local_dir="./fineweb-100BT/",
    allow_patterns="sample/100BT/*"
)



# Or other way

# from datatrove.pipeline.readers import ParquetReader

# # limit determines how many documents will be streamed (remove for all)
# # to fetch a specific dump: hf://datasets/HuggingFaceFW/fineweb/data/CC-MAIN-2024-10
# # replace "data" with "sample/100BT" to use the 100BT sample
# data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb/data", limit=1000) 
# for document in data_reader():
#     # do something with document
#     print(document)

# ###############################    
# # OR for a processing pipeline:
# ###############################

# from datatrove.executor import LocalPipelineExecutor
# from datatrove.pipeline.readers import ParquetReader
# from datatrove.pipeline.filters import LambdaFilter
# from datatrove.pipeline.writers import JsonlWriter

# pipeline_exec = LocalPipelineExecutor(
#     pipeline=[
#         # replace "data/CC-MAIN-2024-10" with "sample/100BT" to use the 100BT sample
#         ParquetReader("hf://datasets/HuggingFaceFW/fineweb/sample/100BT", limit=1000),
#         LambdaFilter(lambda doc: "hugging" in doc.text),
#         JsonlWriter("/home/dyh/The-most-influential-graduation-project/pretrain/datasets/fineweb-100BT")
#     ],
#     tasks=10
# )
# pipeline_exec.run()
