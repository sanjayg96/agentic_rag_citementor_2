This is the version 2 of the CiteMentor.

This time it will include the following additions/enhancements:
1. True agentic routing with LangGraph.
2. Input and output guardrails.
3. Observability dashboard with evals usign RAGAS and also a mechanism to track user queries for which we do not have content in the existing catalog or library. This will act as an indirect user feedback mechanism to understand what are some of the user queries that are not being
addressed with the current library.
4. Support for running this using purely local LLMs. Though evals with RAGAS would use OpenAI's models via API and would need an API key.
Running evals can be controlled via toggle i.e, whether to run evals or not.