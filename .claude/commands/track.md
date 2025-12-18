# /track Command

Create a tracking item (bug, task, idea, or decision) in the appropriate tracking document.

## Tracking System Overview

Tracking items are organized by type in `nimbalyst-local/tracker/`:
- **Bugs** (bugs.md): Issues and defects that need fixing
- **Tasks** (tasks.md): Work items and todos
- **Ideas** (ideas.md): Feature ideas and improvements
- **Decisions** (decisions.md): Architecture and design decisions

## Context-Aware Placement

The command should intelligently choose where to place tracking items:

1. **In current plan document** - If working within a plan file (has `planStatus` frontmatter), add the item to a relevant section (e.g., "Known Issues", "Tasks", "Ideas")
2. **In related plan document** - If the item relates to a specific feature/component, check for a plan document for that feature in the plans directory
3. **In global tracker** - Default to `nimbalyst-local/tracker/[type]s.md` for general items

This keeps related items together for better context and organization.

## Tracking Item Structure

Each tracking item uses inline tracker syntax:

```markdown
- [Brief description] #[type][id:[type]_[ulid] status:to-do priority:medium created:YYYY-MM-DD]
```

### Required Fields

| Field | Format | Description |
|-------|--------|-------------|
| `id` | `[type]_[ulid]` | Unique identifier (bug_, task_, ida_, dec_) |
| `status` | `to-do|in-progress|done` | Current status |
| `priority` | `low|medium|high|critical` | Item priority |
| `created` | `YYYY-MM-DD` | Creation date |

### Optional Fields

| Field | Format | Description |
|-------|--------|-------------|
| `title` | `"Title text"` | Explicit title (if different from line text) |
| `updated` | `YYYY-MM-DDTHH:MM:SS.sssZ` | Last update timestamp (ISO 8601) |
| `assignee` | `username` | Person responsible |

## ULID Generation

Generate a unique ULID (Universally Unique Lexicographically Sortable Identifier):

- **Format**: 26 characters, Base32 encoded
- **Character set**: 0-9, A-Z (excluding I, L, O, U)
- **Structure**: 10 chars timestamp + 16 chars random
- **Example**: `01HQXYZ7890ABCDEF12345`

**ID Prefixes by type**:
- Bugs: `bug_01HQXYZ7890ABCDEF12345`
- Tasks: `task_01HQXYZ7890ABCDEF12345`
- Ideas: `ida_01HQXYZ7890ABCDEF12345`
- Decisions: `dec_01HQXYZ7890ABCDEF12345`

## Examples

### Bug
```markdown
- Login button doesn't work on mobile Safari #bug[id:bug_01HQXYZ7890ABCDEF12345 status:to-do priority:high created:2025-10-24]
```

### Task
```markdown
- Update documentation for API endpoints #task[id:task_01HQXYZ7890ABCDEF12346 status:in-progress priority:medium created:2025-10-24]
```

### Idea
```markdown
- Add dark mode to settings panel #idea[id:ida_01HQXYZ7890ABCDEF12347 status:to-do priority:low created:2025-10-24]
```

### Decision
```markdown
- Use PostgreSQL for data persistence #decision[id:dec_01HQXYZ7890ABCDEF12348 status:done priority:high created:2025-10-20]
```

## Status Values
- `to-do`: Newly created, not yet started
- `in-progress`: Currently being worked on
- `blocked`: Blocked by dependencies or issues
- `done`: Work completed
- `wont-fix`: Decided not to address (bugs/tasks)

## Usage

When the user types `/track [type] [description]`:

Where `[type]` is one of: `bug`, `task`, `idea`, or `decision`

1. **Parse the type** from the command
2. **Generate ULID** for the unique item ID
3. **Determine priority** based on description
4. **Add to appropriate tracker file** in `nimbalyst-local/tracker/[type]s.md`
5. **Confirm** to the user where the item was tracked

**Examples:**
- `/track bug Login fails on mobile Safari`
- `/track task Update API documentation`
- `/track idea Add dark mode support`
- `/track decision Use TypeScript for new modules`

## Priority Guidelines

- **Critical**: System down, data loss, security vulnerability, must-have feature
- **High**: Major feature broken, high-value feature, important decision
- **Medium**: Feature partially broken, nice to have, standard task
- **Low**: Minor issue, cosmetic problem, low-priority enhancement

## Related Commands

- `/plan [description]` - Create a feature plan (see .claude/commands/plan.md)

## Best Practices

- **Always generate new ULIDs** - Never hardcode or reuse IDs
- **Include creation date** - Required for all new items
- **Default to medium priority** - Unless user specifies otherwise
- **Preserve file formatting** - Maintain existing structure
- **Group related items** - Keep items organized by section
- **Update timestamps** - Set `updated` field when modifying items
- **Move completed items** - Move to "Completed" section when done