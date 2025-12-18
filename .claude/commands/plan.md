# /plan Command

Create a new plan document for tracking development work.

## Overview

Plans are structured markdown documents with YAML frontmatter that track features, refactors, bug fixes, and other development work. They provide a single source of truth for what needs to be done, who's responsible, and current progress.

## File Location and Naming

**Location**: `nimbalyst-local/plans/[descriptive-name].md`

**Naming conventions**:
- Use kebab-case: `user-authentication-system.md`
- Be descriptive: The filename should clearly indicate what the plan is about
- Keep it concise: Aim for 2-5 words

## Required YAML Frontmatter

Every plan MUST start with this frontmatter structure:

```yaml
---
planStatus:
  planId: plan-[unique-identifier]  # Use kebab-case
  title: [Plan Title]                # Human-readable title
  status: [status]                   # See Status Values below
  planType: [type]                   # See Plan Types below
  priority: [priority]               # low | medium | high | critical
  owner: [username]                  # Primary owner/assignee
  tags:                              # Keywords for categorization
    - [tag1]
    - [tag2]
  created: "YYYY-MM-DD"             # Creation date (use today's date)
  updated: "YYYY-MM-DDTHH:MM:SS.sssZ"  # Last update timestamp (use current time via new Date().toISOString())
  progress: [0-100]                  # Completion percentage
---
```

**Optional frontmatter fields**:
- `stakeholders`: Array of people interested in this plan
- `dueDate`: Target completion date (YYYY-MM-DD)
- `startDate`: When work began (YYYY-MM-DD)

## Status Values

| Status | When to Use |
|--------|-------------|
| `draft` | Just created, gathering requirements |
| `ready-for-development` | Planning complete, ready to start |
| `in-development` | Actively being implemented |
| `in-review` | Implementation done, awaiting review |
| `completed` | All acceptance criteria met |
| `rejected` | Decided not to pursue |
| `blocked` | Waiting on dependencies |

## Plan Types

| Type | Example |
|------|---------|
| `feature` | Add dark mode, Implement user profiles |
| `bug-fix` | Fix login timeout, Resolve memory leak |
| `refactor` | Migrate to TypeScript, Clean up database |
| `system-design` | Design API architecture, Database schema |
| `research` | Evaluate frameworks, Performance analysis |

## Document Body Structure

After the frontmatter, organize the plan like this:

```markdown
# [Plan Title]

## Goals
- Clear, measurable objectives
- What success looks like
- Key deliverables

## Overview
Brief description of the problem or feature being addressed.

## Implementation Details
Technical details about how this will be implemented.

## Acceptance Criteria
- [ ] Checklist item 1
- [ ] Checklist item 2
- [ ] Checklist item 3
```

## Complete Example

```markdown
---
planStatus:
  planId: plan-user-authentication
  title: User Authentication System
  status: in-development
  planType: feature
  priority: high
  owner: developer
  stakeholders:
    - developer
    - product-team
  tags:
    - authentication
    - security
  created: "2025-10-24"
  updated: "2025-10-24T14:30:00.000Z"
  progress: 45
  startDate: "2025-10-20"
  dueDate: "2025-11-01"
---

# User Authentication System

## Goals
- Implement secure JWT-based authentication
- Support email/password and OAuth (Google, GitHub)
- Add role-based access control (RBAC)

## Overview

The app currently has no authentication. We need a complete auth system with multiple sign-in methods and proper authorization.

## Implementation Details

### Technology Stack
- Passport.js for authentication
- JWT for stateless auth
- Redis for sessions
- bcrypt for password hashing

### API Endpoints
- `POST /auth/register` - User registration
- `POST /auth/login` - Email/password login
- `POST /auth/refresh` - Refresh access token
- `GET /auth/google` - OAuth with Google

## Acceptance Criteria
- [ ] Users can register with email/password
- [ ] Users can log in with email/password
- [ ] OAuth works (Google, GitHub)
- [ ] JWT tokens expire after 15 minutes
- [ ] Role-based permissions work
- [ ] All tests passing
```

## CRITICAL: Timestamp Requirements

When creating a plan:
1. Set `created` to today's date in YYYY-MM-DD format
2. Set `updated` to the CURRENT timestamp using new Date().toISOString() format
3. NEVER use midnight timestamps (00:00:00.000Z) - always use the actual current time

The `updated` field is used to display "last updated" times in the tracker table. Using midnight timestamps will show incorrect "Xh ago" values.

## Usage

When the user types `/plan [description]`:

1. Extract key information from the description
2. Choose appropriate `planType`, `priority`, and `status`
3. Generate unique `planId` from description (kebab-case)
4. Set `created` to today's date, `updated` to current timestamp (use new Date().toISOString())
5. Create file in `nimbalyst-local/plans/` with proper frontmatter
6. Include relevant sections based on plan type

## Related Commands

- `/track [type] [description]` - Track bugs, tasks, ideas, or decisions (see .claude/commands/track.md)
  - Example: `/track bug Login fails on Safari`

## Best Practices

- Keep plans focused (one feature/task per plan)
- Update status and progress regularly
- Use clear, descriptive titles
- Tag appropriately for filtering
- Link related plans in document body
- Break large plans into multiple focused plans