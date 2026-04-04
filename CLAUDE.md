# Claude Code Guidelines

## Key Reference Files for Token Efficiency

### Papers & Research
When asked to check the project's papers:
- **Reference:** `docs/papers.txt` - Contains links and references to papers relevant to this project
- Read this file to understand the theoretical foundation and related work

### Proposed Changes Tracking
**Always check `proposed_changes.txt` at the start of relevant conversations** to see:
- What changes were proposed in previous conversations
- What still needs to be done
- Historical context on why changes were proposed
- This file accumulates proposals over time—do not clear it unless explicitly told to

### Completed Changes
- **Reference:** `changes.txt` - Documents all changes that have been implemented
  - Check this to understand what has been done
  - Reference for historical context on implementation decisions
  - Updated whenever changes are implemented

### Implementation Workflow
When implementing changes:
1. **Start:** Read `proposed_changes.txt` to see all pending proposals (current and past)
2. **Implement:** Make the changes in the codebase
3. **Document:** Update `changes.txt` with a new entry (date + description of what was done)
4. **Clear:** Remove the completed item from `proposed_changes.txt`
5. **Propose New:** If you identify new changes needed, add them to `proposed_changes.txt` with context

### When Making New Proposals
- Add to `proposed_changes.txt` with:
  - Clear description of the change
  - Why it's needed
  - Any dependencies or prerequisites
  - This helps future Claude instances understand the reasoning

---

## Project Context
This is a neurosymbolic MVP project integrating neural networks with symbolic reasoning. The codebase includes neural operator recognition, symbolic knowledge bases, and integration testing.
