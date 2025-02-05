from typing import List, Dict, Union, Optional
from pydantic import BaseModel, ValidationError, Field
from langchain_community.callbacks.manager import get_openai_callback
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import os
import sys
sys.path.append("/workspaces/20241123_RAG_System/src")
import json
import getpass
import dotenv
from dotenv import load_dotenv

class EntityExtractionOutput(BaseModel):
    """
    Pydantic model for the NER task output.
    Defines the structure of the output which we expect from the LLM doing NER.
    """
    PERSON: List[str] = Field(description="List of person names.")
    GPE: List[str] = Field(description="List of entities that are geopolitical entities, i.e. countries, cities, states.")
    ORGANIZATION: List[str] = Field(description="List of entities that are organizations such as companies, non-profit organizations, clubs and institutions.")
    MILITARY_UNIT: List[str] = Field(description="List of entities that are military units such as infantery divisions, battallions, brigades, special forces, etc.")
    DATE: List[str] = Field(description="List of entities that are absolute or relative dates or periods.")
    TIME: List[str] = Field(description="Times smaller than a day.")
    # make this last field optional to avoid validation errors if no values are passed: https://stackoverflow.com/questions/74907839/pydantic-returns-field-required-type-value-error-missing-on-an-optional-fiel
    # and pydantic docu: https://docs.pydantic.dev/2.10/concepts/fields/#default-values
    # and Optional: https://docs.python.org/3/library/typing.html#typing.Optional
    # you need to set 'default' to 'None'! 'Optional'-typing does not suffice!
    OTHER: Optional[Dict[str, List[str]]] = Field(default=None, description="Dictionary that contains any other possible entity type as key and the related list of entities of that type.")

#print(EntityExtractionOutput.model_fields)

class LLMNamedEntityRecognizer:
    json_structure = """{{
        'PERSON': [list of person names],
        'GPE': [list of cities, regions, countries],
        'ORGANIZATION': [list of companies, non-profit organizations, clubs and other institutions],
        'MILITARY_UNIT': [list of military units],
        'DATE': [list of absolute or relative dates or periods],
        'TIME': [list of times smaller than a day],
        'OTHER': {{other_entity_type: [list of entities of that other identified type]}}
    }}"""
    
    def __init__(self, model_name: str="gpt-4o-mini"):
        """
        Initialize the LLMNamedEntityRecognizer with LangChain and an OpenAI API Key.

        Args:
            api_key (str): Your OpenAI API key.
            model_name (str, optional): The LLM you want to use. Defaults to "gpt-4o-mini".
        """
        found_dotenv = dotenv.find_dotenv()
        if found_dotenv:
            load_dotenv()
            OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
        else:
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            max_tokens=3500,
            max_retries=2,
            api_key=OPENAI_API_KEY
        )

    def extract_entities(self, text: str, entity_types: Union[List[str], None]=None) -> EntityExtractionOutput:
        """
        Extract named entities with an LLM from the input text.

        Args:
            text (str): The text to analyze for entities.
            entity_types (Union[List[str], None], optional): A list of specific entity types to extract. Defaults to None. Through 'Union' either a passed list is used or None. If None, all possible entities are extracted.

        Returns:
            EntityExtractionOutput: A structured output is returned as defined with the given pydantic model.
        """
        # Define the entity types in the prompt
        if entity_types:
            entity_types_str = ", ".join(entity_types)
            instruction = f"Extract named entities from the following text and include only these entity types: {entity_types_str}"
        else:
            instruction = f"Extract all possible named entities that you can identify from the following text."
        
        # Prompt structure
        system_message = SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant for named entity recognition. You have to extract named entities from a given text."
        )
        human_message = HumanMessagePromptTemplate.from_template(
            f"{instruction}\n\n"
            """Return the result as a JSON object with the following structure:
            {json_structure}
            Text: {text}""", partial_variables={"json_structure": LLMNamedEntityRecognizer.json_structure}
        )
        entity_extraction_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        with get_openai_callback() as cb:
            # Call the LLM
            structured_entity_extraction_pipe = self.llm.with_structured_output(EntityExtractionOutput)
            structured_entity_extractor = entity_extraction_prompt | structured_entity_extraction_pipe
            response = structured_entity_extractor.invoke({"text": text})
            # print("=======================================================================")
            # print(f"Final Response:\n{response}")
            # print("-----------------------------------------------------------------------")
            # print(f"Total Tokens: {cb.total_tokens}")
            # print(f"Prompt Tokens: {cb.prompt_tokens}")
            # print(f"Completion Tokens: {cb.completion_tokens}")
            # print(f"Total Cost (USD): ${cb.total_cost}")
            # Validate and parse output with Pydantic model
            try:
                entities = response
                json_output = entities.model_dump_json(indent=4)
                with open("entities.json", "w", encoding="utf-8") as file:
                    json.dump(json_output, file, indent=4)
                return entities, json_output
            except ValidationError as e:
                raise ValueError(f"LLM response validation failed: {e}")

# Test
# if __name__=="__main__":
#     ner = LLMNamedEntityRecognizer()
#     text = "Bloomberg L.P. reported yesterday at 2 p.m. in New York at the UN Conference Center that in Kherson the AKHMAT special forces and the 99th Special Operations Battalion recaptured several strategic parts in the Kherson region, Ukraine, and destroyed many vehicles of the Russian 236th infantery division."
#     entity_types = ["PERSON", "DATE", "ORGANIZATION", "TIME", "MILITARY_UNIT", "GPE"]
    
#     try:
#         entities, json_output = ner.extract_entities(text=text, entity_types=entity_types)
#         print(f"Entities in JSON-Format: {json_output}")
        
#         for k in json.loads(json_output).keys():
#             print("Entity {}".format(k))
#     except ValueError as e:
#         print(f"Error, check:{e}")