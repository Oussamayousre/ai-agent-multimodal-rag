import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4


from pymilvus import MilvusClient, DataType
import numpy as np
import concurrent.futures

class FaissVectorStore :
    """
    this class is to create a vectorstore with FAISS 
    """
    def __init__(self, embeddings,chunks,docs_embeddings) : 
        """
        initializing the FAISS vectorstore
        :param embeddings: embedding model
        :param docs_embeddings: document embeddings
        """
        self.embeddings = embeddings
        self.chunks = chunks
        self.docs_embeddings = docs_embeddings
        
        
    def create_vectorstore(self) : 
        """
        Create and store vectors in FAISS vectorstore
        
        """
        index = faiss.IndexFlatL2(len(self.docs_embeddings[0]) )  # dimension des vecteurs
        vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
        uuids = [str(uuid4()) for _ in range(len(self.docs_embeddings))]

        list_vectors = vector_store.add_documents(documents=self.chunks, ids=uuids)

        return list_vectors,vector_store
    


class MilvusColbertRetriever:
    def __init__(self, milvus_client, collection_name, dim=128):
        self.collection_name = collection_name
        self.client = milvus_client
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.load_collection(collection_name)
        self.dim = dim

    def create_collection(self):
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
        
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_fields=True,
        )
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim
        )
        schema.add_field(field_name="seq_id", datatype=DataType.INT16)
        schema.add_field(field_name="doc_id", datatype=DataType.INT64)
        schema.add_field(field_name="doc", datatype=DataType.VARCHAR, max_length=65535)

        self.client.create_collection(
            collection_name=self.collection_name, schema=schema
        )

    def create_index(self):
        self.client.release_collection(collection_name=self.collection_name)
        self.client.drop_index(
            collection_name=self.collection_name, index_name="vector"
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_name="vector_index",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={
                "M": 16,
                "efConstruction": 500,
            },
        )

        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )
    def insert(self, data):
        # Insert ColBERT embeddings and metadata for a document into the collection.
        colbert_vecs = [vec for vec in data["colbert_vecs"]]
        seq_length = len(colbert_vecs)
        doc_ids = [data["doc_id"] for i in range(seq_length)]
        seq_ids = list(range(seq_length))
        docs = [""] * seq_length
        docs[0] = data["filepath"]

        # Insert the data as multiple vectors (one for each sequence) along with the corresponding metadata.
        self.client.insert(
            self.collection_name,
            [
                {
                    "vector": colbert_vecs[i],
                    "seq_id": seq_ids[i],
                    "doc_id": doc_ids[i],
                    "doc": docs[i],
                }
                for i in range(seq_length)
            ],
        
        )
    def search(self, data, topk):
        # First, perform a vector search to find candidate documents
        search_params = {"metric_type": "IP", "params": {}}
        results = self.client.search(
            self.collection_name,
            data,
            limit=int(10),
            output_fields=["vector", "seq_id", "doc_id"],
            search_params=search_params,
        )
        
        # Collect unique document IDs from the results
        doc_ids = set()
        for r_id in range(len(results)):
            for r in range(len(results[r_id])):
                if results[r_id][r].distance >= 0.5 : 
                    doc_ids.add(results[r_id][r]["entity"]["doc_id"])

        scores = []

        if len(doc_ids) == 0 : 
            return scores 

        # Rerank function to calculate MaxSim score for each document
        def rerank_single_doc(doc_id, data, client, collection_name):
            # Retrieve all embeddings for this document
            doc_colbert_vecs = client.query(
                collection_name=collection_name,
                filter=f"doc_id in [{doc_id}]",
                output_fields=["seq_id", "vector", "doc"],
                limit=1000,
            )
            
            # Stack all vectors for this document
            doc_vecs = np.vstack(
                [doc_colbert_vecs[i]["vector"] for i in range(len(doc_colbert_vecs))]
            )
            
            # Calculate MaxSim score: for each query token, find the most similar document token
            # and sum these maximum similarities
            score = np.dot(data, doc_vecs.T).max(1).sum()
            return (score, doc_id)
        # Use parallel processing to rerank documents
        with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
            futures = {executor.submit(rerank_single_doc, doc_id, data, self.client, self.collection_name): doc_id for doc_id in doc_ids}
            for future in concurrent.futures.as_completed(futures):
                score, doc_id = future.result()
                scores.append((score, doc_id))

        # Sort by score and return top-k results
        scores.sort(key=lambda x: x[0], reverse=True)
        if len(scores) >= topk:
            return scores[:topk]
        else:
            return scores

