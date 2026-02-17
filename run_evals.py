from http.client import REQUEST_TIMEOUT
import json
import os
import asyncio
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from langfuse import Langfuse
from src.openai_client import get_client, invoke
from agent_poc import ReActAgent

# Load environment variables
load_dotenv()

# Initialize Langfuse and OpenAI client
langfuse = Langfuse(timeout=120)
client = get_client()

DATASET_NAME = "agent-poc-dataset-v2"
EVAL_DATASET_PATH = "eval_dataset.json"

def load_dataset_items(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r') as f:
        return json.load(f)

def get_or_create_dataset(items: List[Dict[str, Any]]) -> Any:
    # Check if dataset exists, if so return it; else create it.
    try:
        # We use get_dataset to check existence, but it throws if not found?  
        # Actually SDK usually returns object. Let's try to just create it, 
        # normally upsert logic is safer or handled.
        # But for POC we just create. 
        # If it exists, create_dataset might error or return existing?
        # Let's rely on create_datasetitem taking dataset_name.
        langfuse.create_dataset(name=DATASET_NAME)
        print(f"Dataset '{DATASET_NAME}' created.")
    except Exception as e:
        print(f"Dataset might already exist or error: {e}")
    
    # Upsert items
    for i, item in enumerate(items):
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            input=item["input"],
            expected_output=item["expected_output"],
            metadata={"ground_truth": item.get("ground_truth")}
        )
    
    # Return the dataset object for iteration? 
    # Actually get_dataset returns the full object with items if configured?
    # No, usually need to fetch items.
    return langfuse.get_dataset(DATASET_NAME)

def evaluate_response(input_text: str, actual_output: str, expected_output: str) -> float:
    """
    Uses LLM as a Judge to evaluate if the actual output matches the expected output.
    Returns a score between 0.0 and 1.0.
    """
    prompt = f"""
    You are an impartial judge evaluating the performance of an AI agent.
    
    Input Task: {input_text}
    Expected Output (or intended behavior): {expected_output}
    Actual Output: {actual_output}
    
    Does the Actual Output correctly satisfy the Input Task and align with the Expected Output?
    Consider semantic equivalence, not just string matching.
    
    Return ONLY a number between 0.0 (Completely Incorrect) and 1.0 (Perfectly Correct).
    """
    
    try:
        messages = [{"role": "user", "content": prompt}]
        response_text = invoke(client, langfuse, messages, model="gpt-4o-mini")
        score_str = response_text.strip()
        # Simple parsing to ensure we get a float
        score = float(score_str)
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 0.0

def run_evals():
    print("Loading test data...")
    items = load_dataset_items(EVAL_DATASET_PATH)
    
    print("Setting up Langfuse dataset...")
    # 1. Create/Update the dataset in Langfuse
    dataset = get_or_create_dataset(items)
    
    # 2. Initialize Agent
    agent = ReActAgent()
    
    print("Starting evaluation run...")
    
    # Iterate through the items from the fetched dataset
    for item in dataset.items:
        input_data = item.input # The input stored in the dataset
        expected_output = item.expected_output
        
        print(f"\nRunning item: {input_data}")
        
        # 3. Create Trace linked to Dataset Item
        # We use the item.link() context manager to automatically link the trace
        # Wait, the item object from get_dataset() has a link() method?
        # Yes, standard SDK usually provides this.
        # But if it doesn't, we can pass metadata.
        
        # Let's inspect if dataset.items has the link method.
        # Assuming it does based on docs.
        
        # Actually, best practice is:
        # item.get_trace_config() or similar?
        # No, typically:
        # trace = langfuse.trace(...)
        # item.link(trace, run_name="...")
        
        # 3. Create Trace linked to Dataset Item using item.run() context manager
        # This automatically links the trace to the dataset item and handles lifecycle.
        with item.run(
            run_name="evaluation-run3",
            run_metadata={"model": "gpt-4o-mini"},
        ) as trace:
            # Execute Agent
            # The agent.run will now be a child span of this trace if contexts propagate correctly.
            # However, since agent.run creates its own root span via @observe, 
            # we might see two traces or a nested one if using langfuse.trace() vs context.
            # Ideally, agent.run should accept a parent trace or we rely on OTel context.
            # Given agent_poc.py structure, @observe creates a generation/span. 
            # If `trace` here sets the context, @observe should pick it up as parent.
            
            try:
                actual_output = agent.run(input_data)
            except Exception as e:
                actual_output = f"Error: {e}"
                
            # Update trace output
            trace.update(output=actual_output)
            
            # 4. Evaluation
            print(f"  Actual: {actual_output}")
            print(f"  Expected: {expected_output}")
            
            score = evaluate_response(input_data, actual_output, expected_output)
            print(f"  Score: {score}")
            
            # 5. Log Score
            trace.score(
                name="accuracy",
                value=score,
                comment=f"Expected: {expected_output}"
            )
        
        # Sleep to avoid rate limits
        time.sleep(5)

    
    print("\nEvaluation complete. Flushing traces...")
    langfuse.flush()

if __name__ == "__main__":
    run_evals()
