import faiss
import os
import shutil
import gc
import torch
from llama_index.vector_stores import FaissVectorStore
from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index import StorageContext, load_index_from_storage
from llama_index.readers.web import TrafilaturaWebReader


class FaissEmbeddingStorage:
    # Здесь происходит добавления данных в вектор, которые затем будут использоваться в RAG

    def __init__(self, data_dir, dimension, websites_dir=None):
        self.d = dimension
        self.data_dir = data_dir
        self.websites_dir = websites_dir
        self.engine = None
        self.persist_dir = f"{self.data_dir}_vector_embedding"

    def initialize_index(self, force_rewrite=False):
        # Check if the persist directory exists and delete it if force_rewrite is true
        if force_rewrite and os.path.exists(self.persist_dir):
            print("Deleting existing directory for a fresh start.")
            self.delete_persist_dir()

        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            print("Using the persisted value form " + self.persist_dir)
            vector_store = FaissVectorStore.from_persist_dir(self.persist_dir)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=self.persist_dir
            )
            self.index = load_index_from_storage(storage_context=storage_context)
        else:
            print("Generating new values")
            torch.cuda.empty_cache()
            gc.collect()
            if os.path.exists(self.data_dir) and os.listdir(self.data_dir):
                file_metadata = lambda x: {"filename": x}
                documents = SimpleDirectoryReader(
                    self.data_dir, file_metadata=file_metadata, recursive=True,
                    required_exts=[".pdf", ".doc", ".docx", ".txt", ".xml"]
                ).load_data()
            else:
                print("No files found in the directory. Initializing an empty index.")
                documents = []
            if self.websites_dir is not None:
                web_documents = []
                for i in os.listdir(self.websites_dir):
                    with open(os.path.join(self.websites_dir, i), "r") as t:
                        web_documents += TrafilaturaWebReader().load_data(t.read().split("\n"))
                documents += web_documents

            faiss_index = faiss.IndexFlatL2(self.d)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
            index.storage_context.persist(persist_dir=self.persist_dir)
            self.index = index
            torch.cuda.empty_cache()
            gc.collect()

    def delete_persist_dir(self):
        if os.path.exists(self.persist_dir) and os.path.isdir(self.persist_dir):
            try:
                shutil.rmtree(self.persist_dir)
            except Exception as e:
                print(f"Error occurred while deleting directory: {str(e)}")

    def get_engine(self, is_chat_engine, streaming, similarity_top_k):
        if is_chat_engine:
            self.engine = self.index.as_chat_engine(
                chat_mode="condense_question",
                streaming=streaming,
                similarity_top_k=similarity_top_k
            )
        else:
            query_engine = self.index.as_query_engine(
                streaming=streaming,
                similarity_top_k=similarity_top_k,
            )
            self.engine = query_engine
        return self.engine

    def reset_engine(self, engine):
        engine.reset()
