import os
import sys
import argparse
import time
import ingestion, embeddings , vectorstore,test_evaluation
# from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device

from langchain_community.document_loaders import PyPDFLoader,TextLoader
from pymilvus import MilvusClient, DataType

import requests
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from PIL import Image
import os
import json 

from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams,MLLMImage
from deepeval.metrics import GEval


client = MilvusClient(uri="milvus.db")  # For local testing with Milvus Lite
retriever = vectorstore.MilvusColbertRetriever( milvus_client=client,collection_name="colpali")
class SimpleRag : 
    """
    A class that handles simple rag architecutre , ingestion , chnuking ,query retrieval and generation
    """
    def __init__(self, path, chunk_size = 300, chunk_overlap = 10 , n_retrieved = 3  ):
        """
        Intializing the params to Encode PDF , and creating the retriever
        
        :param path: PDF or File path
        :param chunk_size(int): size of each text chunk
        :param chunk_overlap(int): Overlapping between consecutive Chunks
        :param n_retrieved: Number of chunks retrieved
        """
        self.path = path 
        self.chunk_size = chunk_size 
        self.chunk_overlap = chunk_overlap
        self.n_retrieved = n_retrieved


    def test_loader(self) :    
        pdf_lodaer = ingestion.DocumentsLoader(self.path, self.chunk_size, self.chunk_overlap)
        documents = pdf_lodaer.pdfLoader()
        return documents
    def test_chunker(self,document) : 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        texts = text_splitter.split_documents(document)
        return texts
    def test_embedding(self,texts_chunks,model_name = "all-MiniLM-L6-v2" ) : 
        doc_embedder = embeddings.base_embedding(model_name)
        doc_embeddings,hg_embedder = doc_embedder.HgEmbedding(texts_chunks)
        return doc_embeddings,hg_embedder
    def test_vectorstore(self,embeddings,docs_embeddigns,embedd) : 
        vectorstore_creator = vectorstore.FaissVectorStore(embeddings,docs_embeddigns,embedd)
        indexing_result,vector_store= vectorstore_creator.create_vectorstore()
        return indexing_result,vector_store
    def vector_retrieval(self,query,vector_store) : 

        results = vector_store.similarity_search(
            query,
            k=2,
        )
        return [res.page_content for res in results]
    
    def test_ColPali_embedding(self,pdf_path = "/Users/oussamayousr/Documents/ai-agent-multimodal-rag/data/2312.10997v5-2.pdf"
                               ,model_name = "vidore/colpali-v1.2",device = "mps") : 
        """
        test_ColPali_embedding : upload documents , indexing , then saving vectors in milvus
        
        :param pdf_path
        :param model_name: colpali model name
        :param device: used device 
        """
        img_loader = ingestion.DocumentsLoader(pdf_path)
        img_path, img_base_path = img_loader.pdf_img_loader()
        doc_embedder = embeddings.base_embedding(model_name = model_name,device = device)
        model , processor = doc_embedder.ColPAli()
        # Process document images

        images = [Image.open(img_base_path+f"/{name}") for name in os.listdir(img_base_path)]
        print("images paths",images)
        dataloader = DataLoader(
            dataset=ListDataset[str](images),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: processor.process_images(x),
        )
        print("dataloader",dataloader)

        document_embeddings = []
        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
                embeddings_doc = model(**batch_doc)
                print("type embeddings_doc",type(embeddings_doc))
            document_embeddings.extend(list(torch.unbind(embeddings_doc.to(device))))



        # Create and set up the Milvus collection

        retriever.create_collection()
        retriever.create_index()

        # Insert embeddings into Milvus
        filepaths = [img_base_path+f"/{name}" for name in os.listdir(img_base_path)]
        for i in range(len(filepaths)):
            data = {
                "colbert_vecs": document_embeddings[i].float().cpu().numpy(),
                "doc_id": i,
                "filepath": filepaths[i],
            }
            retriever.insert(data)
    def test_colpali_rag(self,query,model_name = "vidore/colpali-v1.2",device = "mps", model_url = "http://127.0.0.1:8000/generate" ) :
        doc_embedder = embeddings.base_embedding(model_name = model_name,device = device)
        print("Enter your query :")
        input_query = input()
        model , processor = doc_embedder.ColPAli()

        # Process queries
        queries = [
           input_query,
        ]
        process_start_time  = time.time()
        model_input = processor.process_queries(queries).to(model.device)
        print("process queries time : ",time.time() - process_start_time)
        embedding_start_time  = time.time()
        embeddings_query = model(**model_input)
        print("embedding queries time : ",time.time() - embedding_start_time)
        # Search for each query
        search_start_time = time.time()
        for i, query in enumerate(queries):
            query_embedding = embeddings_query[i].cpu().float().numpy()
            results = retriever.search(query_embedding, topk=3)
        print("search queries time : ",time.time() - search_start_time)



        base_path = "/Users/oussamayousr/Documents/ai-agent-multimodal-rag/data/colipali_data/"
        
        files  = [("files",(os.path.basename(base_path+f"page_{doc_id+1}.png"), open(base_path+f"page_{doc_id+1}.png", 'rb'), 'image/png')) for  _, doc_id in results ] # or appropriate image MIME type
            

        data = {
        "token": queries[0]  # This goes in data, not files
         }

        actual_output = ''
        with requests.post(model_url,  data=data, files=files, stream=True) as response:
            for line in response.iter_lines():
                if not line:
                    actual_output+= line

                    continue
                print(line.decode('utf-8'))
        rag_evaluation  = test_evaluation.Rag_Eval()
        retrieval_context=f"""        
        {MLLMImage(url="./image_folders", local=True)} 
    """

        rag_evaluation.Eval_rag(metrics_list,input,actual_output,retrieval_context)
        
        

    def test_correctness():
        correctness_metric = GEval(
            name="Correctness",
            criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=0.5
        )
        test_case = LLMTestCase(
            input="I have a persistent cough and fever. Should I be worried?",
            # Replace this with the actual output from your LLM application
            actual_output="A persistent cough and fever could be a viral infection or something more serious. See a doctor if symptoms worsen or don't improve in a few days.",
            expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."
        )
        assert_test(test_case, [correctness_metric])

    

  
                



        






    

if __name__ == '__main__':
    simple_rag = SimpleRag(
        path = "/Users/oussamayousr/Documents/ai-agent-multimodal-rag/data/2312.10997v5-2.pdf"    
    )


    text = simple_rag.test_loader()
    chunks = simple_rag.test_chunker(text)
    embedd,hg_embedder = simple_rag.test_embedding(chunks,model_name = "all-MiniLM-L6-v2" )
    emb = embeddings.base_embedding(model_name = "all-MiniLM-L6-v2")
    indexing_results,vector_store = simple_rag.test_vectorstore(hg_embedder,chunks,embedd)
    query = "LangChain provides abstractions to make working with LLMs easy"

    retrieval_results = simple_rag.vector_retrieval(query,vector_store)

    # simple_rag.test_ColPali_embedding(pdf_path = "/Users/oussamayousr/Documents/ai-agent-multimodal-rag/data/2312.10997v5-2.pdf"
    #                            ,model_name = "vidore/colpali-v1.2",device = "mps")
    simple_rag.test_colpali_rag(query = "nothing",model_name = "vidore/colpali-v1.2",device = "mps")
    
    




        
        

    

# deepeval key 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


        