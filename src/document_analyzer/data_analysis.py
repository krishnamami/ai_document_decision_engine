import os
import sys
from utils.model_loader import Model_Loader
from logger.custom_logger import CustomLogger
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
from model.models import *
from langchain_core.output_parsers import JsonOutputToolsParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.prompts import *
from prompt.prompt_library import PROMPT_REGISTRY 


class DocumentAnalyzer:
    """Analyzes documents using a pre-trained model.
    Automatically logs all actions and supports session-based organization
    """
    def __init__(self):
        self.log=CustomLogger().get_logger(__name__)
        try:
            self.loader=Model_Loader()
            self.llm=self.loader.load_llm()
            self.parser=JsonOutputParser(pydantic_object=Metadata)
            self.fixing_parser=OutputFixingParser.from_llm(self.llm,self.parser)
            self.prompt=PROMPT_REGISTRY["document_analysis"]
            log.info("DocumentAnalyzer initialized successfully")
        except Exception as e:
            self.log.error("Error initializing DocumentAnalyzer",error=str(e))
            raise DocumentPortalException("Error initiating DocumentAnalyzer", e) from e
        
    def analyze_document(self,document_text:str)->dict:
        """Analyzes the given document text and returns metadata as a dictionary."""
        try:
            chain=self.prompt |self.llm | self.fixing_parser
            log.info("Meta-data analysis chain initialized")

            response=chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "document_text": document_text
            })
            log.info("Metadata extracted successfully",keys=list(response.keys()))
            return response
        except Exception as e:
            self.log.error("Metadata analysis failed",error=str(e))
            raise DocumentPortalException("Metadata extraction failed", sys) 