# PROJECT RULES — Always Respect These

## RULE #1: Behavioral Guidelines

Before any implementation, follow these rules in order:

**Think Before Coding**
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present all of them.
- If a simpler approach exists, say so.
- If unclear, stop and ask.

**Simplicity First** — Minimum code that solves the problem. No speculative features, abstractions, or error handling for impossible scenarios. If it can be 50 lines instead of 200, rewrite it.

**Surgical Changes** — Touch only what you must. Don't improve adjacent code, refactor things that aren't broken, or change formatting. Remove only imports/variables YOUR changes made unused.

**Goal-Driven Execution** — Define verifiable success criteria. Multi-step tasks get a brief plan with verification checks. Loop until criteria are met.

---

## INSTALLED SKILLS (load on demand via `skill` tool)

### Global (~/.config/opencode/skills/)
| Skill | When to load |
|---|---|
| `cloudflare` | Building on Cloudflare Workers, D1, R2, KV, AI, Durable Objects, WAF |
| `composio-cli` | Integrating SaaS tools (GitHub, Linear, Slack, Figma, Stripe etc.) via CLI |
| `mcp-builder` | Building MCP servers — protocol, SDK setup, evals |
| `stop-slop` | Editing/rewriting docs, READMEs, or prose — removes AI writing tells |
| `vault-daydream` | Mining Obsidian vault notes for non-obvious connections |
| `vercel-react-best-practices` | Writing/reviewing/refactoring React or Next.js code |

### Agent (~/.agents/skills/)
| Skill | When to load |
|---|---|
| `skill-creator` | Creating, testing, benchmarking, or improving SKILL.md files |
| `frontend-design` | Designing UI mockups, landing pages, dashboards, or visual interfaces |
| `playwright` | Browser automation, testing web apps, taking screenshots |
| `plan` | Task breakdown and implementation planning |
| `code-review` | Reviewing code diffs for correctness and quality |
| `comprehensive-review` | Full code review with parallel subagents |
| `research` | Codebase exploration and pattern searching |
| `init` | Setting up repo documentation and contributor guidelines |

### Plugin — Obra Superpowers
Plugin `superpowers@git+https://github.com/obra/superpowers.git` provides full-agentic development methodology (brainstorming, write-plan, execute-plan, TDD, debugging, code review). Ask: "Tell me about your superpowers"
