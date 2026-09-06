# dynavec — explainer video script & storyboard

A ~90-second narrated walkthrough. Record it as a voiceover over a screen capture of
`explainer.html`, or as slides. Tone: clear, confident, developer-to-developer.

**How to produce it**
1. Open `explainer.html` full-screen and screen-record it (Loom / OBS / QuickTime) while reading the narration below — the scene timing is already tuned to this script.
2. Or record slides/Keynote using the same beats.
3. Upload to YouTube or Loom, then paste the embed URL into `explainer.html` (see the comment block near the bottom of that file) to replace the animation with your video.

Total run time target: **85–95 seconds**.

---

### Scene 1 — Title (0:00–0:06)
**On screen:** dynavec logo · "The serverless vector database that lives in *your* AWS account."
**Narration:** "This is dynavec — a serverless vector database that runs entirely inside your own AWS account. Here's the 90-second tour."

### Scene 2 — The problem (0:06–0:14)
**On screen:** three cards — Pinecone / OpenSearch / Qdrant·Weaviate·Milvus.
**Narration:** "Every vector database today forces a trade-off. Pinecone is managed — so your data leaves your account. OpenSearch is powerful but costs money even when idle. Qdrant, Weaviate, and Milvus mean you run and scale the servers yourself. And at a billion vectors, any of these can cost tens of thousands of dollars a month."

### Scene 3 — The goal (0:14–0:22)
**On screen:** three bullets — serverless / in your account / billions, no servers.
**Narration:** "So we asked a simple question: what if your vector database was just… AWS? Serverless, so you pay only when you use it. In your own account and region, for compliance. Scaling to billions of vectors — with no servers to run."

### Scene 4 — How it works (0:22–0:31)
**On screen:** Embed → S3 Vectors → DynamoDB → Results flow.
**Narration:** "dynavec fuses two AWS primitives you already trust. Amazon S3 Vectors does the billion-scale nearest-neighbor search. DynamoDB stores your documents and hydrates them in single-digit milliseconds. dynavec joins them by a shared key — and adds filtering, reranking, and more on top."

### Scene 5 — The proof (0:31–0:39)
**On screen:** $469/mo vs $80k+/mo.
**Narration:** "The result? At a billion 1536-dimension vectors, dynavec runs about four hundred and sixty-nine dollars a month — versus tens of thousands for a managed cluster. Because your storage is priced like S3, not RAM."

### Scene 6 — What's inside (0:39–0:47)
**On screen:** feature list.
**Narration:** "And it's batteries-included: bring-your-own-key embedders, metric reranking, namespaces for multi-tenant RAG, a knowledge graph, query caching, MCP-based ingestion, and adapters for LangChain, LlamaIndex, and CrewAI."

### Scene 7 — Contribute (0:47–0:56)
**On screen:** pip install · 100+ issues · good first issues.
**Narration:** "dynavec is open source under Apache 2.0 — and we'd love your help. Install it with pip and try it in your own account. There are over a hundred issues, labeled by area and difficulty, including plenty of good first issues."

### Scene 8 — Call to action (0:56–1:03)
**On screen:** github.com/codeforstartups/dynavec.
**Narration:** "Star the repo, read the docs, and pick an issue. Let's build the vector database that stays in your account. Thanks for watching."

---

**Lower-thirds / captions to overlay:** repo URL, `pip install dynavec`, and the docs link (https://codeforstartups.github.io/dynavec/docs/).
