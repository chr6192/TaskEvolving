这个目录下放供给tool 相关的内容，要求能够支持符合openai tool schema的异步高并发tool excute

tools:
```json
        Example tool_schema:
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Searches for relevant information based on queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_list": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of search queries"
                            }
                        },
                        "required": ["query_list"]
                    }
                }
            }
        """
```

```python
class OpenAIFunctionPropertySchema(BaseModel):
    """The schema of a parameter in OpenAI format."""

    type: str
    description: str | None = None
    enum: list[str] | None = None


class OpenAIFunctionParametersSchema(BaseModel):
    """The schema of parameters in OpenAI format."""

    type: str
    properties: dict[str, OpenAIFunctionPropertySchema]
    required: list[str]


class OpenAIFunctionSchema(BaseModel):
    """The schema of a function in OpenAI format."""

    name: str
    description: str
    parameters: OpenAIFunctionParametersSchema
    strict: bool = False


class OpenAIFunctionToolSchema(BaseModel):
    """The schema of a tool in OpenAI format."""

    type: str
    function: OpenAIFunctionSchema
```