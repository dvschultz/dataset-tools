# /track-bug Command

Track a bug using Nimbalyst's inline tracker syntax.

## Overview

The `/track-bug` command creates bug tracking items using a lightweight inline syntax. Bugs can be tracked in dedicated tracker files or directly within plan documents for context-aware organization.

## Context-Aware Bug Tracking

The command automatically determines the best location for the bug:

### 1. In Current Plan Document
If you're working on a plan document (has `planStatus` frontmatter):
- Bug is added to the current plan file
- Added in a relevant section (e.g., "Bugs", "Known Issues", "Problems")
- If no such section exists, creates "## Known Issues" section

### 2. In Related Feature Plan
If the bug is related to a specific feature/component:
- Checks for a plan document for that feature in `nimbalyst-local/plans/`
- If found, adds the bug there for context

### 3. In Global Bug Tracker
Otherwise (general bug or no specific context):
- Adds to `nimbalyst-local/tracker/bugs.md`
- Creates the file with proper structure if it doesn't exist

## Bug Tracker Syntax

Use inline tracker syntax with `#bug` prefix:

```markdown
- [Brief bug description] #bug[id:bug_[ulid] status:to-do priority:medium created:YYYY-MM-DD]
```

### Required Fields

| Field | Format | Description |
|-------|--------|-------------|
| `id` | `bug_[ulid]` | Unique identifier (26-char ULID) |
| `status` | `to-do|in-progress|done` | Current status |
| `priority` | `low|medium|high|critical` | Bug severity |
| `created` | `YYYY-MM-DD` | Creation date |

### Optional Fields

| Field | Format | Description |
|-------|--------|-------------|
| `title` | `"Title text"` | Explicit title (if different from line text) |
| `updated` | `YYYY-MM-DDTHH:MM:SS.sssZ` | Last update timestamp (ISO 8601) |

## ULID Generation

Generate a unique ULID (Universally Unique Lexicographically Sortable Identifier):

- **Format**: 26 characters, Base32 encoded
- **Character set**: 0-9, A-Z (excluding I, L, O, U)
- **Structure**: 10 chars timestamp + 16 chars random
- **Example**: `01HQXYZ7890ABCDEF12345`
- **Full bug ID**: `bug_01HQXYZ7890ABCDEF12345`

**Why ULID?**
- Lexicographically sortable (sorts by creation time)
- No central coordination needed
- URL-safe and case-insensitive
- More compact than UUIDs

## Examples

### Simple Bug
```markdown
- Login button doesn't work on mobile Safari #bug[id:bug_01HQXYZ7890ABCDEF12345 status:to-do priority:high created:2025-10-24]
```

### Bug with Explicit Title
```markdown
- Safari mobile login issue #bug[id:bug_01HQXYZ7890ABCDEF12346 status:in-progress priority:high created:2025-10-24 title:"Mobile Safari Login Failure"]
```

### Bug with Update Timestamp
```markdown
- API timeout on large requests #bug[id:bug_01HQXYZ7890ABCDEF12347 status:to-do priority:critical created:2025-10-24 updated:2025-10-24T14:30:00.000Z]
```

### Completed Bug
```markdown
- Memory leak in image loader #bug[id:bug_01HQXYZ7890ABCDEF12348 status:done priority:high created:2025-10-20 updated:2025-10-24T16:00:00.000Z]
```

## Bug Tracker File Structure

If creating `nimbalyst-local/tracker/bugs.md`, use this template:

```markdown
# Bugs

## Active Bugs

- [New and in-progress bugs with #bug syntax]

## Completed Bugs

- [Completed bugs with status:done]
```

## Usage Workflow

When the user types `/track-bug [description]`:

1. **Extract bug details** from the user's description
2. **Determine location** based on context (plan, related feature, or global tracker)
3. **Generate ULID** for the unique bug ID
4. **Create bug entry** with proper inline syntax
5. **Add to appropriate section** in the target file
6. **Confirm** to the user where the bug was tracked

## Priority Guidelines

Choose priority based on impact:

- **Critical**: System down, data loss, security vulnerability
- **High**: Major feature broken, affects many users
- **Medium**: Feature partially broken, workaround exists
- **Low**: Minor issue, cosmetic problem, edge case

## Status Transitions

Typical bug lifecycle:

```
to-do → in-progress → done
         ↓
      blocked (if stuck)
```

## Related Commands

- `/plan [description]` - Create a feature plan (see .claude/commands/plan.md)
- `/track-idea [description]` - Track an idea (see .claude/commands/track-idea.md)

## Best Practices

- **Always generate new ULIDs** - Never hardcode or reuse IDs
- **Include creation date** - Required for all new bugs
- **Default to medium priority** - Unless user specifies otherwise
- **Preserve file formatting** - Maintain existing structure and styling
- **Group related bugs** - Keep bugs near related content in plans
- **Update timestamps** - Set `updated` field when modifying bugs
- **Move completed bugs** - Move to "Completed" section when done