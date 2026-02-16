import os
import datetime
import json
import re
from typing import Dict, Any, List

from dotenv import load_dotenv
from google import genai
from langfuse import Langfuse, observe

# Load environment variables
load_dotenv()

# Initialize Langfuse
# It automatically picks up LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST from env
langfuse = Langfuse(
    timeout=120
)

# Configure Google GenAI
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Tools ---

@observe(as_type="generation")
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression."""
    try:
        # unsafe eval for poc purposes only
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Error evaluating expression: {e}"

@observe(as_type="generation")
def get_current_time(args: str = "") -> str:
    """Returns the current local time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@observe(as_type="generation")
def reverse_text(text: str) -> str:
    """Reverses the input text."""
    return text[::-1]

@observe(as_type="generation")
def mock_search(query: str) -> str:
    """Mock search engine that returns dummy results."""
    return f"Found 3 results for '{query}': 1. Result A, 2. Result B, 3. Result C"

TOOLS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "reverse_text": reverse_text,
    "mock_search": mock_search
}

TOOL_DESCRIPTIONS = """
1. calculator(expression: str) -> str: Useful for performing math calculations.
2. get_current_time() -> str: Useful for getting the current local time.
3. reverse_text(text: str) -> str: Useful for reversing a string.
4. mock_search(query: str) -> str: Useful for searching information.
"""

# --- Agent ---

class ReActAgent:
    def __init__(self):
        self.system_prompt = f"""
You are a helpful AI assistant that uses tools to answer questions.
You utilize a ReAct (Reasoning and Acting) loop.

Available Tools:
{TOOL_DESCRIPTIONS}

Format your response exactly as follows:

Thought: <your reasoning>
Action: <tool_name>
Action Input: <input_string>
Observation: <result_of_tool>
... (repeat Thought/Action/Observation as needed)
Final Answer: <the final answer to the user>

If you already know the answer or don't need tools, go straight to Final Answer.
"""

    @observe(as_type="generation")
    def call_llm(self, prompt: str) -> str:
        """Wrapper to call Gemini and trace it."""
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            langfuse.update_current_generation(
                usage_details={
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                },
                model="gemini-2.0-flash"
            )
            
        return response.text

    @observe(name="agent_run")
    def run(self, user_input: str) -> str:
        """
        Main execution loop.
        Traced as a single transaction in Langfuse.
        """
        conversation_history = self.system_prompt + f"\nUser: {user_input}\n"
        max_steps = 5
        
        # We can update the current trace with input metadata
        langfuse.update_current_trace(
            input=user_input,
            metadata={"agent_type": "ReAct", "model": "gemini-2.0-flash"}
        )

        for step in range(max_steps):
            # 1. GENERATE THOUGHT & ACTION
            response_text = self.call_llm(conversation_history)
            
            # Use regex to find Action and Action Input
            thought_match = re.search(r"Thought:(.*?)(Action:|Final Answer:|$)", response_text, re.DOTALL)
            action_match = re.search(r"Action:\s*(.*)", response_text)
            input_match = re.search(r"Action Input:\s*(.*)", response_text)
            
            thought = thought_match.group(1).strip() if thought_match else ""
            
            # Print for local debugging
            print(f"\n[Step {step+1}]")
            print(f"Thoughts: {thought}")

            # Check for Final Answer
            if "Final Answer:" in response_text:
                final_answer = response_text.split("Final Answer:")[-1].strip()
                print(f"Final Answer: {final_answer}")
                
                # --- EVALUATION STEP ---
                # Programmatically scoring the trace
                langfuse.score_current_trace(
                    name="agent_confidence",
                    value=0.95, # Mock score
                    comment="Agent successfully reached a final answer."
                )
                
                return final_answer
            
            # Execute Tool
            if action_match and input_match:
                tool_name = action_match.group(1).strip()
                tool_input = input_match.group(1).strip()
                
                print(f"Invoking Tool: {tool_name} with input: {tool_input}")
                
                if tool_name in TOOLS:

                    try:
                        # Arguments might need adjustment depending on tool signature
                        # For simplicity, we pass the single string input
                        tool_result = TOOLS[tool_name](tool_input)
                    except Exception as e:
                        tool_result = f"Error executing tool: {e}"
                else:
                    tool_result = f"Error: Tool '{tool_name}' not found."
                
                print(f"Observation: {tool_result}")
                
                # Append to history
                conversation_history += f"{response_text}\nObservation: {tool_result}\n"
            else:
                # If structure is broken, append and hope LLM fixes it or stops
                conversation_history += f"{response_text}\nObservation: Invalid format. Please use Thought, Action, Action Input.\n"

        return "Agent timed out or failed to parse response." 

if __name__ == "__main__":
    agent = ReActAgent()
    
    # Only run this if the script is executed directly
    print("--- Starting Agent PoC ---")
    user_query = input("User Query: ")
    
    result = agent.run(user_query)
    print(f"\nResult: {result}")
    
    # Flush Langfuse callback to ensure traces are sent
    langfuse.flush()
