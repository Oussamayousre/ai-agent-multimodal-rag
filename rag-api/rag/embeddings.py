from langchain_huggingface import HuggingFaceEmbeddings
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from PIL import Image
import os




class base_embedding: 
    """
    the logic of this class is to implement all the possible embedding option : 
    local with ray , Local with hugginface/langchain , and with APIs
    """
    def __init__(self,model_name,device = "mps"):

        self.model_name = model_name
        # self.documents = documents
        self.device = device 

    def HgEmbedding(self,chunks) : 
        
        # docs_embeddings = embeddings.embed_documents([chunk.page_content for chunk in chunks])
        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in chunks:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)
            else : 
                print("Duplicate found and removed:", doc.page_content)
        embeddings_model = HuggingFaceEmbeddings(model_name=self.model_name)

        doc_embedding = embeddings_model.embed_documents(chunk.page_content for chunk in chunks)
        return doc_embedding,embeddings_model
    
    def ColPAli(self) : 
        device = get_torch_device(self.device) 

        model = ColPali.from_pretrained(
        self.model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        ).eval()

        processor = ColPaliProcessor.from_pretrained(self.model_name)
        return model,processor


