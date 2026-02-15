from typing import List, Dict, Optional, Any
from openai import OpenAI
import json
import os 
import json
import wikipedia
from typing import Dict, List, Any
from openai import OpenAI


os.environ["OPENAI_API_KEY"] = "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"

tools = [{
    "type": "function",
    "function": {
        "name": "search_wikipedia",
        "description": """Search Wikipedia and return a concise summary.
Returns the first three sentences of the most relevant article.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic to search for on Wikipedia"
                }
            },
            "required": ["query"],
            "additionalProperties": False
        },
        "strict": True
    }
    ,
    "type": "function",
    "function": {
        "name": "search_rag",
        "description": """search for the context in the vectorDB , if the question if purely about RAGs""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "the user query to search for similarities in VectorDB"
                }
            },
            "required": ["query"],
            "additionalProperties": False
        },
        "strict": True
    }
}]


def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia and return a concise summary.
    Handles disambiguation and missing pages gracefully.
    """
    try:
        # Try to get the most relevant page summary
        summary = wikipedia.summary(
            query,
            sentences=3,
            auto_suggest=True,
            redirect=True
        )

        return json.dumps({
            "success": True,
            "summary": summary,
            "url": wikipedia.page(query).url
        })

    except wikipedia.DisambiguationError as e:
        # Handle multiple matching pages
        return json.dumps({
            "error": "Disambiguation error",
            "options": e.options[:5],  # List first 5 options
            "message": "Topic is ambiguous. Please be more specific."
        })

    except wikipedia.PageError:
        return json.dumps({
            "error": "Page not found",
            "message": f"No Wikipedia article found for: {query}"
        })

    except Exception as e:
        return json.dumps({
            "error": "Unexpected error",
            "message": str(e)
        })
from rag import simple_rag

Colpali_search = simple_rag.SimpleRag(
        path = "/Users/oussamayousr/Documents/ai-agent-multimodal-rag/data/2312.10997v5-2.pdf"    
    )
def search_rag(query: str) -> str:
    """
    Search in rag vector DV  and return the context and the query.
    Handles disambiguation and missing pages gracefully.
    """
    try:
        # Try to get the most relevant page summary
        results = Colpali_search.test_colpali_rag(
            query
            
        )

        return json.dumps({
            "vlm_result": results,
        })

    except Exception as e:
        return json.dumps({
            "error": "Unexpected error",
            "message": str(e)
        })

class Agent:
    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize an AI Agent with optional system prompt.

       Args:
            system_prompt: Initial instructions for the AI
        """
        # Initialize OpenAI client - expects OPENAI_API_KEY in environment
        self.client = OpenAI()

        # Initialize conversation history
        self.messages = []

        # Set up system prompt if provided, otherwise use default
        default_prompt = """You are a helpful AI assistant with access to a Rag 
        and Wikipedia. Follow these rules:
        1. When asked about RAGs informations ,always search in vectordb first,
        2. For general knowledge questions, use Wikipedia
        4. Always mention your source of information from wikipedia
        5. If a tool returns an error, explain the error to the user clearly
        6. send the same user query to RAG function ,  make it longer.

        """

        self.messages.append({
            "role": "system",
            "content": system_prompt or default_prompt
        })

    def execute_tool(self, tool_call: Any) -> str:
        """
        Execute a tool based on the LLM's decision.

        Args:
           tool_call: The function call object from OpenAI's API

        Returns:
            str: JSON-formatted result of the tool execution
        """
        try:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # Execute the appropriate tool. Add more here as needed.
            # if function_name == "query_database":
            #     result = query_database(function_args["query"])
            if function_name == "search_wikipedia":
                result = search_wikipedia(function_args["query"])
            if function_name == "search_rag":
                print("query", function_args["query"])
                result = search_rag(function_args["query"])
            else:
                result = json.dumps({
                    "error": f"Unknown tool: {function_name}"
                })

            return result

        except json.JSONDecodeError:
            return json.dumps({
                "error": "Failed to parse tool arguments"
            })
        except Exception as e:
            return json.dumps({
                "error": f"Tool execution failed: {str(e)}"
            })

    def process_query(self, user_input: str) -> str:
        """
        Process a user query through the AI agent.

        Args:
            user_input: The user's question or command

        Returns:
            str: The agent's response
        """
        # Add user input to conversation history
        self.messages.append({
            "role": "user",
            "content": user_input
        })

        try:
            max_iterations = 5
            current_iteration = 0

            while current_iteration < max_iterations:  # Limit to 5 iterations
                current_iteration += 1
                completion = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=self.messages,
                    tools=tools,  # Global tools list from Step 1
                    tool_choice="auto"  # Let the model decide when to use tools
                )

                response_message = completion.choices[0].message
                print("tools call response_message",response_message,response_message.tool_calls)

                # If no tool calls, we're done
                if not response_message.tool_calls:
                    self.messages.append(response_message)
                    return response_message.content

                # Add the model's thinking to conversation history
                self.messages.append(response_message)

                # Process all tool calls
                for tool_call in response_message.tool_calls:
                    try:
                        result = self.execute_tool(tool_call)
                        print("Tool executed......")
                    except Exception as e:
                        print("Execution failed......")
                        result = json.dumps({
                            "error": f"Tool execution failed: {str(e)}"
                        })

                    # print(f"Tool result custom: {result}")

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })
                    # print("Messages:", self.messages)

            # If we've reached max iterations, return a message indicating this
            max_iterations_message = {
                "role": "assistant",
                "content": "I've reached the maximum number of tool calls (5) without finding a complete answer. Here's what I know so far: " + response_message.content
            }
            self.messages.append(max_iterations_message)
            return max_iterations_message["content"]

        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            self.messages.append({
                "role": "assistant",
                "content": error_message
            })
            return error_message

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.

        Returns:
            List[Dict[str, str]]: The conversation history
        """
        return self.messages
    
from fastapi import FastAPI,HTTPException,File, UploadFile,Form
from typing import Annotated

app = FastAPI()
agent = Agent()
@app.post("/generate") 
async def generate(self, token: Annotated[str, Form()]):



        try:
            return agent.process_query(token)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))





# if __name__ == '__main__':
#     agent = Agent()
#     agent.process_query("can you explain everything about RAG")
