# ai-agent-multimodal-rag

ai-agent-multimodal-rag/
│
├── apps/                  # Runtime applications
├── agents/                # Agent definitions & tools
├── rag/                   # RAG pipeline (clean & isolated)
├── llm/                   # LLM providers abstraction
├── infra/                 # Terraform & cloud IaC
├── eval/                  # Evaluation & benchmarks
├── observability/         # Logging, metrics, tracing
├── ui/                    # Simple UI (optional)
├── configs/               # Centralized config
├── scripts/               # Dev & ops scripts
├── docker/                # Dockerfiles
├── tests/                 # Tests
│
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── Makefile


1 - deploy llm + visual + multi-model + vocal models using ray test different option (quantized / distilled  / Mini ... )