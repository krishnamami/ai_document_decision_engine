import os
import sys
import fitz
import uuid
from datetime import datetime
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

class DocumentHandler:
    """Handles PDF saving and reading operations
    Automatically logs all actions and supports session-based organization
    """
    def __init__(self,data_dir=None,session_id=None):
        try:
            self.log=CustomLogger().get_logger(__name__)
            self.data_dir=data_dir or os.getenv("DATA_STORAGE_PATH",os.path.join(os.getcwd(),"data","document_analysis"))
            self.session_id=session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            self.session_path=os.path.join(self.data_dir,self.session_id)
            os.makedirs(self.session_path,exist_ok=True)
            self.log.info("PDFHandler initialized",session_id=self.session_id,session_path=self.session_path)
        except Exception as e:
            self.log.error("Error initializing PDFHandler",error=str(e))
            raise DocumentPortalException("Error initiating DocumentHandler", e) from e

        
    def save_pdf(self,uploaded_file):
        try:
            filename=os.path.basename(uploaded_file.name)
            if not filename.lower().endswith(".pdf"):
                raise DocumentPortalException("Invalid file type. Only PDF files are supported.")
            save_path=os.path.join(self.session_path,filename)
            with open(save_path,"wb") as f:
                f.write(uploaded_file.read())   
            self.log.info("PDF saved successfully",file=filename,save_path=save_path,session_id=self.session_id)
            return save_path
        except Exception as e:
            self.log.error("Error saving PDF",error=str(e))
            raise DocumentPortalException("Error saving PDF", e) from e 

    def read_pdf(self,pdf_path):
        try:
            text_chunks=[]
            with fitz.open(pdf_path) as doc:
                for page_num,page in enumerate(doc,start=1):
                    text_chunks.append(f"\n----Page {page_num} ---\n{page.get_text() }")
                text="\n".join(text_chunks)
                return text
        except Exception as e:
            self.log.error("Error reading PDF",error=str(e))
            raise DocumentPortalException("Error reading PDF", e) from e
if __name__ == "__main__":
    from pathlib import Path
    from io import BytesIO
    class DummyFile:
        def __init__(self,filepath):
            self.filepath=filepath
            self.name=Path(filepath).name
        def getbuffer(self):
            return open(self.filepath,"rb").read() 
        def read(self):
            return self.getbuffer()
    pdf_path=r"C:\\Users\\bkgou\\OneDrive\\Documents\\document_portal_rag_application\\data\\document_analysis\\NIPS-2017-attention-is-all-you-need-Paper.pdf"
        
    dummy_pdf=DummyFile(pdf_path)
    handler=DocumentHandler(session_id="test_session15")
    try:
       save_path = handler.save_pdf(dummy_pdf)
       content= handler.read_pdf(save_path)
       #print(f"PDF Content Length: {len(content)} characters")
       print(content[:500])  # Print first 500 characters
    except Exception as e:
        print(f"Error: {e}")
    
   
  
        


   
