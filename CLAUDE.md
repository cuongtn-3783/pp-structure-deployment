# pp-structure-deployment

## Project Description
Create a project setup and code to deploy PPStructureV3 on a GPU-enabled server using Docker.

Please follow these strict requirements:
- **Environment & Language**: Use Python 3.12 syntax and uv for environment management.
- **Code Quality:** Use ruff for linting, pyright for type-checking, and pytest for testing.
- **API Framework:** Build a FastAPI application to receive and process requests. All prediction parameters must be accepted via the API request body.
- **Docker Deployment:** Configure the Dockerfile to support GPU execution. Expose and map port 2603:2603, and ensure the port is defined using an ENV variable inside the Dockerfile
- **Model Initialization & Languages:** Download and initialize the PPStructureV3 models immediately upon container startup (e.g., using a FastAPI lifespan event or an entrypoint script), not lazily upon receiving the first request. Ensure the model is configured to support both English and Japanese languages.

## Output Token Optimization Rules

1. **Answer first.** Lead with the direct answer; add context only after.
2. **No restating.** Never repeat or paraphrase the question.
3. **Minimum sufficient words.** Cut filler, hedging, repetition, and decorative phrasing. Every token must earn its place — but never at the cost of accuracy or completeness.
4. **Structure over prose.** Use bullets, tables, equations, or pseudocode when they improve clarity; otherwise plain prose.
5. **Label claim types.** Separate facts, assumptions, and recommendations explicitly.
6. **Keep the exceptions.** Brevity must not drop caveats, edge cases, or important exceptions.
7. **Length cap.** Keep any single paragraph or section under ~200 words; split or restructure if longer.
8. **Code exception.** Code prioritizes clarity and readability over token count — meaningful names, comments where needed, no golfing. Only surrounding explanation is minimized.
9. **Reasoning exception.** For complex reasoning (math, logic, debugging, multi-step analysis), show key reasoning steps before the answer. Rule 1 yields to this: reasoning may precede the answer when correctness depends on it.
10. **Precedence.** When rules conflict: accuracy > completeness (caveats/exceptions) > reasoning steps > brevity. Brevity applies to style, never to substance
