"""Use dynavec as an OpenAI Assistant or Function Calling tool.

This adapter allows an OpenAI Assistant or Chat Completions model to perform
vector retrieval over your Dynavec knowledge base directly without paying for
OpenAI's hosted vector store files.

Requirements:
    pip install "dynavec[sentence-transformers]"
    python examples/openai_assistant_tool.py
"""

import json
from types import SimpleNamespace

from dynavec import Document, Dynavec, DynavecConfig
from dynavec.embeddings import SentenceTransformerEmbedder
from dynavec.integrations.tools import as_openai_tool


def main() -> None:
    # 1. Initialize Dynavec with a local/in-memory configuration
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")

    cfg = DynavecConfig(
        vector_bucket="dynavec-demo",
        index="openai-assistant-docs",
        table="dynavec_assistant_kb",
        dimension=embedder.dimension,
        region="us-east-1",
        auto_provision=True,
    )

    db = Dynavec(cfg, embedder=embedder)

    # 2. Populate knowledge base
    db.upsert(
        [
            Document(
                id="doc-1",
                text="Dynavec is a serverless vector database running on AWS DynamoDB and S3 Vectors.",
            ),
            Document(
                id="doc-2",
                text="Dynavec provides sub-50ms hybrid keyword and vector retrieval at zero idle cost.",
            ),
            Document(
                id="doc-3",
                text="OpenAI Assistants can call custom external retrieval tools during run loops.",
            ),
        ],
        namespace="docs",
    )

    # 3. Create the OpenAI Assistant Tool adapter
    tool = as_openai_tool(
        db,
        name="dynavec_search",
        description="Search documentation and return relevant knowledge base passages.",
        namespace="docs",
        top_k=2,
    )

    print("=== OpenAI Tool Schema ===")
    print(json.dumps(tool.schema, indent=2))
    print()

    # 4. Simulate an Assistant Run loop requesting a tool call
    # In real usage with OpenAI SDK:
    #   run = client.beta.threads.runs.create_and_poll(...)
    #   if run.status == "requires_action":
    #       tool_outputs = tool.submit_tool_outputs(run.required_action.submit_tool_outputs.tool_calls)
    #       client.beta.threads.runs.submit_tool_outputs(..., tool_outputs=tool_outputs)

    simulated_tool_call = SimpleNamespace(
        id="call_mock_12345",
        type="function",
        function=SimpleNamespace(
            name="dynavec_search",
            arguments=json.dumps({"query": "serverless vector database cost"}),
        ),
    )

    print("=== Simulating Tool Execution for Assistant Run ===")
    outputs = tool.submit_tool_outputs([simulated_tool_call])
    print("Submitted Tool Output:")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
