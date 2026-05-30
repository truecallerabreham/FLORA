---
name: code-review
description: Review code changes for bugs, security issues, and improvements
triggers: review, code review, check code, review PR
---

# Code Review Skill

## Review Checklist

### 1. Correctness
- Does the code do what it claims?
- Are edge cases handled (null, empty, bounds)?
- Is error handling appropriate?

### 2. Security
- Input validation present?
- No SQL injection, XSS, command injection?
- Secrets not hardcoded?
- Permissions checked?

### 3. Performance
- No N+1 queries?
- Appropriate data structures?
- Unnecessary loops or allocations?

### 4. Maintainability
- Clear naming?
- Functions do one thing?
- No magic numbers?
- Tests included?

## Review Format

For each issue found:

```
**[SEVERITY] file:line - Brief description**

Problem: What's wrong
Impact: Why it matters
Suggestion: How to fix
```

Severities: CRITICAL, HIGH, MEDIUM, LOW, NIT

## Workflow

1. Read the diff or changed files
2. Understand the intent (PR description, commit messages)
3. Check each file against the checklist
4. Summarize findings with severity levels
