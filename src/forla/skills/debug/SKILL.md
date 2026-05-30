---
name: debug
description: Systematic approach to debugging errors and unexpected behavior
triggers: debug, error, bug, fix, not working, broken
---

# Debug Skill

## Debugging Process

### 1. Reproduce
- Get the exact error message
- Find the minimal steps to trigger it
- Note the environment (Python version, OS, dependencies)

### 2. Locate
- Read the full stack trace (bottom = where it crashed, top = root cause)
- Find the relevant file and line
- Read surrounding code for context

### 3. Understand
- What was the code trying to do?
- What input caused the failure?
- What assumption was violated?

### 4. Fix
- Make the minimal change that fixes the issue
- Don't refactor while debugging
- Add a test that would have caught this

### 5. Verify
- Run the failing case again
- Run the full test suite
- Check for similar patterns elsewhere

## Common Patterns

**NoneType errors**: Something returned None unexpectedly. Check the call chain.

**Import errors**: Missing dependency or circular import. Check requirements and import order.

**Key/Index errors**: Accessing something that doesn't exist. Add existence checks.

**Type errors**: Wrong type passed. Check function signatures and callers.

## Workflow

1. `grep` for the error message in codebase
2. Read the stack trace bottom-to-top
3. Read the failing function
4. Add logging/print if needed to trace values
5. Fix and verify
