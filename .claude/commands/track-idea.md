# /track-idea Command

Track a feature idea using Nimbalyst's inline tracker syntax.

## Overview

The `/track-idea` command creates idea tracking items for feature requests, improvements, and enhancements. Ideas can be tracked in dedicated files or within plan documents for context-aware organization.

## Context-Aware Idea Tracking

The command automatically determines the best location for the idea:

### 1. In Current Plan Document
If you're working on a plan document (has `planStatus` frontmatter):
- Idea is added to the current plan file
- Added in a relevant section (e.g., "Ideas", "Future Enhancements", "Improvements")
- If no such section exists, creates "## Future Ideas" section

### 2. In Related Feature Plan
If the idea is related to a specific feature/component:
- Checks for a plan document for that feature in `nimbalyst-local/plans/`
- If found, adds the idea there for context

### 3. In Global Ideas Tracker
Otherwise (general idea or no specific context):
- Adds to `nimbalyst-local/tracker/ideas.md`
- Creates the file with proper structure if it doesn't exist

## Idea Tracker Syntax

Use inline tracker syntax with `#idea` prefix:

```markdown
- [Brief idea description] #idea[id:ida_[ulid] status:to-do priority:medium created:YYYY-MM-DD]
```

### Required Fields

| Field | Format | Description |
|-------|--------|-------------|
| `id` | `ida_[ulid]` | Unique identifier (26-char ULID) |
| `status` | `to-do|in-progress|done` | Current status |
| `priority` | `low|medium|high|critical` | Idea importance |
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
- **Full idea ID**: `ida_01HQXYZ7890ABCDEF12345`

**Why ULID?**
- Lexicographically sortable (sorts by creation time)
- No central coordination needed
- URL-safe and case-insensitive
- More compact than UUIDs

## Examples

### Simple Idea
```markdown
- Add dark mode to settings panel #idea[id:ida_01HQXYZ7890ABCDEF12345 status:to-do priority:medium created:2025-10-24]
```

### Idea with Explicit Title
```markdown
- Dark mode settings #idea[id:ida_01HQXYZ7890ABCDEF12346 status:in-progress priority:high created:2025-10-24 title:"Dark Mode Theme Switcher"]
```

### Idea with Update Timestamp
```markdown
- Add keyboard shortcuts for common actions #idea[id:ida_01HQXYZ7890ABCDEF12347 status:to-do priority:low created:2025-10-24 updated:2025-10-24T14:30:00.000Z]
```

### Implemented Idea
```markdown
- Auto-save draft messages #idea[id:ida_01HQXYZ7890ABCDEF12348 status:done priority:high created:2025-10-20 updated:2025-10-24T16:00:00.000Z]
```

## Ideas Tracker File Structure

If creating `nimbalyst-local/tracker/ideas.md`, use this template:

```markdown
# Ideas

## Active Ideas

- [New and in-progress ideas with #idea syntax]

## Implemented Ideas

- [Implemented ideas with status:done]
```

## Usage Workflow

When the user types `/track-idea [description]`:

1. **Extract idea details** from the user's description
2. **Determine location** based on context (plan, related feature, or global tracker)
3. **Generate ULID** for the unique idea ID
4. **Create idea entry** with proper inline syntax
5. **Add to appropriate section** in the target file
6. **Confirm** to the user where the idea was tracked

## Priority Guidelines

Choose priority based on value and effort:

- **Critical**: Must-have feature, competitive necessity
- **High**: High-value feature, significant user benefit
- **Medium**: Nice to have, moderate value
- **Low**: Minor enhancement, low priority

## Status Transitions

Typical idea lifecycle:

```
to-do → in-progress → done
   ↓
rejected (if decided not to implement)
```

## Related Commands

- `/plan [description]` - Create a feature plan (see .claude/commands/plan.md)
- `/track-bug [description]` - Track a bug (see .claude/commands/track-bug.md)

## Best Practices

- **Always generate new ULIDs** - Never hardcode or reuse IDs
- **Include creation date** - Required for all new ideas
- **Default to medium priority** - Unless user specifies otherwise
- **Preserve file formatting** - Maintain existing structure and styling
- **Group related ideas** - Keep ideas near related content in plans
- **Update timestamps** - Set `updated` field when modifying ideas
- **Move implemented ideas** - Move to "Implemented" section when done
- **Convert to plans** - Promote high-value ideas to full plan documents