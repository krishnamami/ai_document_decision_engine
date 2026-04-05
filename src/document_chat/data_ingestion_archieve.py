import sys
import uuid
from pathlib import Path
from typing import Iterable, List

# Keep your environment's loader/vectorstore imports if required.
# If these fail in your env, switch to the alternative comments below.
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# Common LangChain imports (change if your environment differs)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from utils.model_loader import Model_Loader


class SingleDocIngestor:
    def __init__(self, data_dir: str = "data/single_document_chat", faiss_dir: str = "faiss_index"):
        try:
            self.log = CustomLogger().get_logger(__name__)

            self.data_dir = Path(data_dir).resolve()
            self.data_dir.mkdir(parents=True, exist_ok=True)

            self.faiss_dir = Path(faiss_dir).resolve()
            self.faiss_dir.mkdir(parents=True, exist_ok=True)

            # consistent naming for the model loader
            self.model_loader = Model_Loader()

            # configure text splitter (tune chunk_size/overlap to your needs)
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)

            self.log.info(f"SingleDocIngestor initialized: data_dir={self.data_dir}, faiss_dir={self.faiss_dir}")

            # DO NOT instantiate FAISS() here. Use FAISS.from_documents(...) or FAISS.load_local(...)
        except Exception as e:
            self.log.error(f"Failed to initialize SingleDocIngestor: {e}", exc_info=True)
            raise DocumentPortalException("Initialization failed for SingleDocIngestor", sys)

    def ingest_files(self, uploaded_files: Iterable):
        """
        uploaded_files: iterable of file-like objects (supporting .read(), optionally .name)
        returns: a retriever built from ingested PDFs
        """
        documents = []

        for uploaded_file in uploaded_files:
            temp_path = None
            try:
                # defensively rewind uploaded stream if possible
                if hasattr(uploaded_file, "seek"):
                    try:
                        uploaded_file.seek(0)
                    except Exception:
                        pass

                # collision-safe filename
                unique_filename = f"session_{uuid.uuid4().hex}.pdf"
                temp_path = self.data_dir / unique_filename

                # write upload to disk
                with open(temp_path, "wb") as fh:
                    fh.write(uploaded_file.read())

                self.log.info(f"Saved upload to {temp_path} (orig name: {getattr(uploaded_file, 'name', 'unknown')})")

                # create loader per file (requires file path)
                loader = PyPDFLoader(str(temp_path))
                docs = loader.load()

                if docs:
                    documents.extend(docs)
                else:
                    self.log.warning(f"No pages extracted from {temp_path}")

            except Exception as e:
                # isolate failure; continue with next file
                self.log.error(f"Failed to ingest {getattr(uploaded_file, 'name', temp_path)}: {e}", exc_info=True)
            finally:
                # always try to remove temporary file
                if temp_path and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception as ex:
                        self.log.warning(f"Failed to delete temp file {temp_path}: {ex}")

        self.log.info(f"Ingestion finished. Pages loaded: {len(documents)}")

        if not documents:
            raise DocumentPortalException("No valid documents were ingested", sys)

        return self._create_retriever(documents)

    def _create_retriever(self, documents: List):
        try:
            # split into chunks
            splitter = self.text_splitter
            chunks = splitter.split_documents(documents)
            self.log.info(f"Documents split into chunks: {len(chunks)}")

            # load embeddings from your model loader (ensure method name matches your implementation)
            # Prefer a consistent method name in Model_Loader such as `load_embeddings`
            if hasattr(self.model_loader, "load_embeddings"):
                embeddings = self.model_loader.load_embeddings()
            elif hasattr(self.model_loader, "load_Embeddings"):
                embeddings = self.model_loader.load_Embeddings()
            else:
                raise DocumentPortalException("Model_Loader has no recognized embeddings loader", sys)

            if embeddings is None:
                raise DocumentPortalException("Embeddings loader returned None", sys)

            # build FAISS vectorstore from chunks
            vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
            vectorstore.save_local(str(self.faiss_dir))
            self.log.info(f"FAISS vector store created and saved at {self.faiss_dir}")

            # return retriever
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            self.log.info("Retriever created from FAISS vector store")
            return retriever

        except DocumentPortalException:
            raise
        except Exception as e:
            self.log.error(f"Retriever creation failed: {e}", exc_info=True)
            raise DocumentPortalException("Error creating retriever from documents", sys)