# Self-hosted Langfuse (astrag)

1. From the repo root: `docker compose -f compose.yaml up -d` — see [`../compose/README.md`](../compose/README.md) and [`langfuse.docker-compose.yml`](../compose/langfuse.docker-compose.yml) (Langfuse 3 stack).
2. In the Langfuse UI, create a project and copy **public/secret** keys into your environment (see `.env.example` in this folder).
3. Set `LANGFUSE_HOST` to the API base (often `http://localhost:3000`) before `astrag run` with `langfuse_enabled: true` or `LANGFUSE_ENABLED=true`.

Upstream reference: [Langfuse self-hosting](https://langfuse.com/docs/deployment/self-host).

The `astrag` CLI uses the LangChain `CallbackHandler`; keep the `langfuse` Python package major version aligned with the server.
