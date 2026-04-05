# # Test code for document ingestion and analysis using a PDFHandler and DocumentAnalyzer



import sys
from pathlib import Path
from langchain_community.vectorstores import FAISS
from src.document_chat.data_ingestion_archieve import SingleDocIngestor
from src.document_chat.retrieval import ConversationalRAG
from utils.model_loader import Model_Loader

FAISS_INDEX_PATH = Path("faiss_index").resolve()

def test_conversational_rag_on_pdf(pdf_path: str, question: str):
    model_loader = Model_Loader()

    try:
        # Ensure pdf exists early
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Step 1: Load or build FAISS retriever
        retriever = None
        if FAISS_INDEX_PATH.exists() and any(FAISS_INDEX_PATH.iterdir()):
            print(f"Found faiss index at {FAISS_INDEX_PATH}. Attempting to load...")
            try:
                embeddings = model_loader.load_Embeddings()
                vectorstores = FAISS.load_local(str(FAISS_INDEX_PATH), embeddings,
                                               allow_dangerous_deserialization=True)
                retriever = vectorstores.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                print("FAISS index loaded successfully.")
            except Exception as e:
                print(f"Failed to load existing FAISS index ({FAISS_INDEX_PATH}): {e}")
                print("Will rebuild the index from the PDF.")
        else:
            print(f"No FAISS index found at {FAISS_INDEX_PATH}. Will build a new one.")

        if retriever is None:
            # Rebuild index from PDF
            with open(pdf_path, "rb") as f:
                uploaded_files = [f]
                ingestor = SingleDocIngestor()
                retriever = ingestor.ingest_files(uploaded_files)
            print("Rebuilt FAISS index and created retriever.")

        # Step 2: Run RAG
        print("Running Conversational RAG ...")
        session_id = "test_conv_rag"
        rag = ConversationalRAG(session_id=session_id, retriever=retriever)
        response = rag.invoke(question)
        print(f"\nQuestion: {question}\nAnswer: {response}")

    except Exception as e:
        # Do not sys.exit inside library code; raise to caller or print with context for ad-hoc tests
        print(f"Error during Conversational RAG test: {e}", file=sys.stderr)
        raise

#if __name__ == "__main__":
    pdf_path = r"C:\Users\bkgou\OneDrive\Documents\document_portal_rag_application\data\single_document_chat\NIPS-2017-attention-is-all-you-need-Paper.pdf"
    question = "Summarize it?"

    try:
        test_conversational_rag_on_pdf(pdf_path, question)
    except Exception as exc:
        # Provide a short exit code for scripts; still surface stack trace above
        sys.exit(1)


    
import sys
from pathlib import Path
from src.multi_document_chat.data_ingestion import DocumentIngestor
from src.multi_document_chat.retrieval import ConversationalRAG
from src.multi_document_chat.data_ingestion import DocumentIngestor





def test_document_ingestion_and_rag():
     try:
         test_files = [
             "data\\multi_document_chat\\LongReportV1.pdf",
             "data\\multi_document_chat\\LongReportV2.pdf",
             "data\\multi_document_chat\\NIPS-2017-attention-is-all-you-need-Paper.pdf"
             
             
         ]
        
         uploaded_files = []
        
         for file_path in test_files:
             if Path(file_path).exists():
                 uploaded_files.append(open(file_path, "rb"))
             else:
                 print(f"File does not exist: {file_path}")
                
         if not uploaded_files:
             print("No valid files to upload.")
             sys.exit(1)
            
         ingestor = DocumentIngestor()
        
         retriever = ingestor.ingest_files(uploaded_files)
        
         for f in uploaded_files:
             f.close()
                
         session_id = "test_multi_doc_chat"
        
         rag = ConversationalRAG(session_id=session_id, retriever=retriever)
        
         question = "what is ther in LongReportV1"
        
         answer=rag.invoke(question)
        
         print("\n Question:", question)
        
         print("Answer:", answer)
        
         if not uploaded_files:
             print("No valid files to upload.")
             sys.exit(1)
            
     except Exception as e:
         print(f"Test failed: {str(e)}")
         sys.exit(1)
        
if __name__ == "__main__":
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    import os
    os.environ["GOOGLE_API_KEY"] = "AIzaSyD24mj4vy9jOYRYLsYO7KENdeuLZqAOBWw"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    docs = ["chunk 1 text", "chunk 2 text", "chunk 3 text"]
    vectors = []
    for doc in docs:
        try:
            vec = embeddings.embed_query(doc)
            vectors.append(vec)
            print(vectors)
        except Exception as e:
            print("Failed to embed chunk:", e)
    test_document_ingestion_and_rag()
   