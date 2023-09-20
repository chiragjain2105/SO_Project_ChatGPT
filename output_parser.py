from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class TagProblem(BaseModel):
    tags: str = Field(description="problem description",min_length=0)

    def to_dict(self):
        return {"tags":self.tags}


TagProblem_parser:PydanticOutputParser = PydanticOutputParser(pydantic_object=TagProblem)