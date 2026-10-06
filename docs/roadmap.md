# dynavec — Agent Orchestration Roadmap

*This is a living plan, not a fixed schedule. The order below is a proposal and will shift as work lands. Tracked under the ecosystem epic (#276).*

## 1. Vision

dynavec began as a vector database and is growing into a full agent stack that runs entirely inside your own cloud account. Four projects cover the lifecycle of an AI application: what it knows, what it does, what happened when it ran, and how good the results were.

| Project  | Tagline            | What it covers                                                                                   | Status |
|----------|---------------------|---------------------------------------------------------------------------------------------------|--------|
| dynavec  | Remember            | Vector database (S3 Vectors + DynamoDB), caching, knowledge graph, hybrid retrieval, integrations | Available |
| dynaflow | Build & run         | Agent orchestration engine (nodes, edges, state, branches, loops, retries) with a visual builder  | In progress — chat models shipped (#285); engine next. Epic #273 |
| dynalogs | Observe             | Per-step traces, structured logs, run history, metrics dashboard, kept in your own account        | Early version: telemetry dashboard. Epic #274 |
| dynaevals| Measure & improve   | Retrieval metrics, LLM-judged quality, test datasets, CI regression checks                         | Partly available — retrieval metrics, faithfulness/answer relevance, trend tracking. Epic #275 |

## 2. How the pieces fit together

dynaflow runs the agent; its retriever/memory steps read and write dynavec, with run state saved to DynamoDB for resumability. dynalogs collects what each step did into a single searchable view of the run. dynaevals scores those same runs and can fail a CI build on a quality regression. Evaluation results feed back into the agent's prompts, steps, and retrieval settings.

## 3. What you can use today

Chat models (dynaflow), one interface across providers, and retrieval evaluation (dynaevals) with recall, MRR and nDCG helpers are already shipped in the dynavec package. See the docs site for full code examples.

## 4. Where the engine is headed (pseudocode, not yet implemented)

The dynaflow orchestration engine (#277 to #281) is still being designed. A graph definition is expected to look roughly like: define a Graph with a state schema, add nodes for retrieve, plan, act and observe steps, connect them with edges and a conditional edge back to planning, then compile with a DynamoDB checkpointer. This is for discussion, not a committed API.

## 5. Proposed order of work

| Area | What's included | Issues |
|------|------------------|--------|
| Engine core | Graph/state runtime, branches, loops, parallel steps, retries, DynamoDB-backed run state, snapshot resume/replay | #277–#281 |
| Nodes and models | Typed node schema, built-in/custom nodes, chat models (done), prompt templates, output parsers | #282–#286 |
| Agents | Tool registry with MCP connectors, tool-calling loop/planner, multi-agent handoffs, conversation memory | #287–#290 |
| Visual builder | Drag-and-drop canvas, code-to-canvas conversion, live run watching, step-through debugging, CLI | #291–#295 |
| Observability | Per-step traces, structured logs, run history, run metrics | #299–#302 |
| Evaluation | More RAG metrics, test datasets, regression runs with CI checks, scores alongside traces | #303–#306 |
| Production & multi-cloud | Human-approval pauses, long-running workflows, one-command serverless deploy, GCP/Azure backends | #296–#298, #307–#308 |

## 6. Contributing

Pick an unassigned issue, comment to get assigned, and wait for a maintainer to confirm before starting. See CONTRIBUTING.md for the full workflow.
