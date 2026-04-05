import sys
from dotenv import load_dotenv
import pandas as pd
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.models import SummaryResponse
from prompt.prompt_library import PROMPT_REGISTRY
from utils.model_loader import Model_Loader
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser

class DocumentCompareLLM:
    def __init__(self):
        load_dotenv()
        self.log = CustomLogger().get_logger(__name__)
        self.loader = Model_Loader()
        self.llm = self.loader.load_llm()

        self.base_parser = JsonOutputParser(pydantic_object=SummaryResponse)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.base_parser, llm=self.llm)

        # Your registry uses string keys
        self.prompt = PROMPT_REGISTRY["document_comparison"]

        # Use the fixing parser
        self.chain = self.prompt | self.llm | self.fixing_parser

        self.log.info("DocumentCompareLLM initialized successfully")

    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        try:
            inputs = {
                "combined_docs": combined_docs,
                "format_instruction": self.base_parser.get_format_instructions(),
            }

            self.log.info("Invoking document comparison LLM chain")
            response = self.chain.invoke(inputs)
            self.log.info("Chain invoked successfully", response_preview=str(response)[:200])

            return self._format_response(response)

        except Exception as e:
            self.log.exception("Error in compare_documents")
            raise DocumentPortalException("An error occurred while comparing documents.", sys) from e

    def _format_response(self, response_parsed) -> pd.DataFrame:
        try:
            # Case 1: {"items": [...]}
            if isinstance(response_parsed, dict) and "items" in response_parsed:
                return pd.DataFrame(response_parsed["items"])

            # Case 2: {"Page 1": {"Changes": ...}, ...}
            if isinstance(response_parsed, dict):
                rows = []
                for page, payload in response_parsed.items():
                    if isinstance(payload, dict):
                        changes = payload.get("changes", payload.get("Changes", payload))
                    else:
                        changes = payload
                    rows.append({"page": page, "changes": changes})
                return pd.DataFrame(rows)

            # Case 3: list of dicts
            if isinstance(response_parsed, list):
                return pd.DataFrame(response_parsed)

            # Fallback
            return pd.DataFrame([{"page": "unknown", "changes": response_parsed}])

        except Exception as e:
            self.log.exception("Error formatting response into DataFrame")
            raise DocumentPortalException("Error formatting response", sys) from e