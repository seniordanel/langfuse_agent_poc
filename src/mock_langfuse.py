import uuid
import json
import time

class Langfuse:
    def __init__(self, **kwargs):
        print("[MockLangfuse] Initialized")
        self.traces = {}

    def trace(self, name=None, **kwargs):
        trace_id = str(uuid.uuid4())
        print(f"[MockLangfuse] Created trace {trace_id} (name={name})")
        self.traces[trace_id] = {"name": name, "scores": [], "metadata": kwargs}
        return MockTrace(trace_id, self)

    def score(self, trace_id, name, value, comment=None):
        print(f"[MockLangfuse] Scored trace {trace_id}: {name} = {value} ({comment})")
        if trace_id in self.traces:
            self.traces[trace_id]["scores"].append({"name": name, "value": value, "comment": comment})

    def flush(self):
        print("[MockLangfuse] Flushed traces.")

class MockTrace:
    def __init__(self, trace_id, client):
        self.trace_id = trace_id
        self.client = client

    def score(self, name, value, comment=None):
        self.client.score(self.trace_id, name, value, comment)
    
    def update(self, **kwargs):
        print(f"[MockLangfuse] Updated trace {self.trace_id}: {kwargs}")

from langchain_core.callbacks import BaseCallbackHandler

class CallbackHandler(BaseCallbackHandler):
    def __init__(self, session_id=None, tags=None, user_id=None, **kwargs):
        super().__init__()
        self.session_id = session_id
        self.tags = tags
        self.user_id = user_id
        self.trace_id = str(uuid.uuid4())
        print(f"[MockCallbackHandler] Initialized for session {session_id} with trace_id {self.trace_id}")

    def get_trace_id(self):
        return self.trace_id

    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"[MockCallbackHandler] Chain start: {serialized}")

    def on_chain_end(self, outputs, **kwargs):
        print(f"[MockCallbackHandler] Chain end")

    def on_chain_error(self, error, **kwargs):
        print(f"[MockCallbackHandler] Chain error: {error}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"[MockCallbackHandler] Tool start: {serialized}")

    def on_tool_end(self, output, **kwargs):
        print(f"[MockCallbackHandler] Tool end")

    def on_tool_error(self, error, **kwargs):
        print(f"[MockCallbackHandler] Tool error: {error}")
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"[MockCallbackHandler] LLM start")
        
    def on_llm_end(self, response, **kwargs):
        print(f"[MockCallbackHandler] LLM end")
        
    def on_llm_error(self, error, **kwargs):
        print(f"[MockCallbackHandler] LLM error: {error}")

# Global instance
# langfuse = Langfuse()
