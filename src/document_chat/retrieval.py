import sys
import os
import time
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from utils.model_loader import Model_Loader
from exception.custom_exception import DocumentPortalException
from logger import GLOBAL_LOGGER as log
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType


class RetryEmbeddings(Embeddings):
    """
    Retry-enabled wrapper for embeddings providers.
    Compatible with LangChain Embeddings interface and older callable paths.
    """

    def __init__(self, base_embeddings, max_retries: int = 5, base_delay: float = 1.0):
        self.base_embeddings = base_embeddings
        self.max_retries = max_retries
        self.base_delay = base_delay

    def _is_retriable(self, error: Exception) -> bool:
        msg = str(error).upper()
        retriable_markers = [
            "500",
            "503",
            "429",
            "INTERNAL",
            "UNAVAILABLE",
            "RESOURCE_EXHAUSTED",
            "TIMEOUT",
        ]
        return any(marker in msg for marker in retriable_markers)

    def _sleep(self, attempt: int):
        time.sleep(self.base_delay * (2 ** attempt))

    def embed_query(self, text: str) -> List[float]:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                log.info(
                    "Embedding query",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    text_preview=text[:200],
                )
                return self.base_embeddings.embed_query(text)
            except Exception as e:
                last_error = e
                log.warning(
                    "Query embedding failed",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e),
                )
                if not self._is_retriable(e) or attempt == self.max_retries - 1:
                    raise
                self._sleep(attempt)
        raise last_error

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                log.info(
                    "Embedding documents",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    document_count=len(texts),
                )
                return self.base_embeddings.embed_documents(texts)
            except Exception as e:
                last_error = e
                log.warning(
                    "Document embedding failed",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e),
                )
                if not self._is_retriable(e) or attempt == self.max_retries - 1:
                    raise
                self._sleep(attempt)
        raise last_error

    def __call__(self, text: str) -> List[float]:
        """
        Compatibility for code paths that still treat embedding_function as callable.
        """
        return self.embed_query(text)


class ConversationalRAG:
    """
    LCEL-based Conversational RAG with lazy retriever initialization.

    Usage:
        rag = ConversationalRAG(session_id="abc")
        rag.load_retriever_from_faiss(index_path="faiss_index/abc", k=5, index_name="index")
        answer = rag.invoke("What is this document about?", chat_history=[])
    """

    def __init__(self, session_id: Optional[str], retriever=None):
        try:
            self.session_id = session_id

            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXT_QA.value
            ]

            self.retriever = retriever
            self.chain = None

            if self.retriever is not None:
                self._build_lcel_chain()

            log.info("ConversationalRAG initialized", session_id=self.session_id)

        except Exception as e:
            log.error("Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentPortalException("Initialization error in ConversationalRAG", sys)

    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = 5,
        index_name: str = "index",
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Load FAISS vectorstore from disk and build retriever + LCEL chain.
        """
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

            base_embeddings = Model_Loader().load_embeddings()
            embeddings = RetryEmbeddings(base_embeddings)

            vectorstore = FAISS.load_local(
                folder_path=index_path,
                embeddings=embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,
            )

            if search_kwargs is None:
                search_kwargs = {"k": k}

            self.retriever = vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs,
            )

            self._build_lcel_chain()

            log.info(
                "FAISS retriever loaded successfully",
                index_path=index_path,
                index_name=index_name,
                k=k,
                session_id=self.session_id,
            )
            return self.retriever

        except Exception as e:
            log.error("Failed to load retriever from FAISS", error=str(e))
            raise DocumentPortalException("Loading error in ConversationalRAG", sys)

    def invoke(
        self,
        user_input: str,
        chat_history: Optional[List[BaseMessage]] = None,
    ) -> str:
        """
        Invoke the LCEL pipeline.
        """
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke().",
                    sys,
                )

            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}

            answer = self.chain.invoke(payload)

            if not answer:
                log.warning(
                    "No answer generated",
                    user_input=user_input,
                    session_id=self.session_id,
                )
                return "no answer generated."

            log.info(
                "Chain invoked successfully",
                session_id=self.session_id,
                user_input=user_input,
                answer_preview=str(answer)[:150],
            )
            return answer

        except Exception as e:
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException("Invocation error in ConversationalRAG", sys)

    def _load_llm(self):
        try:
            llm = Model_Loader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            log.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error in ConversationalRAG", sys)

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _log_rewritten_query(self, query: str) -> str:
        log.info(
            "Rewritten query generated",
            session_id=self.session_id,
            query_preview=query[:500],
        )
        return query

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain", sys)

            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            retrieve_docs = (
                question_rewriter
                | self._log_rewritten_query
                | self.retriever
                | self._format_docs
            )

            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            log.info("LCEL graph built successfully", session_id=self.session_id)

        except Exception as e:
            log.error(
                "Failed to build LCEL chain",
                error=str(e),
                session_id=self.session_id,
            )
            raise DocumentPortalException("Failed to build LCEL chain", sys)