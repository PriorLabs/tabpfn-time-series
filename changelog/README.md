# Changelog Fragments

This directory contains changelog "fragments" — small files describing
the user-visible changes in each PR. At release time, the
`release-create-pr.yml` workflow runs `towncrier build`, which assembles
fragments into `CHANGELOG.md` and deletes them from this directory.

Writing the entry in the PR that makes the change means the release notes
are already written when a release is cut, and two PRs touching the same
release never conflict on `CHANGELOG.md`.

## How to add a changelog entry

1. Create a file named `<PR_NUMBER>.<category>.md` in this directory.
2. Write one short, user-friendly sentence describing the change.

```bash
# Using towncrier create (recommended — validates the category)
towncrier create 712.added.md --content "Add support for custom quantile levels"

# Or manually
echo "Add support for custom quantile levels" > changelog/712.added.md
```

## Categories

| Filename suffix | Section in CHANGELOG | When to use |
|---|---|---|
| `<PR>.breaking.md` | Breaking Changes | Backward-incompatible changes (renamed/removed APIs, dependency contract changes that affect users) |
| `<PR>.added.md` | Added | New features |
| `<PR>.changed.md` | Changed | Modifications to existing functionality |
| `<PR>.fixed.md` | Fixed | Bug fixes |
| `<PR>.deprecated.md` | Deprecated | Marking features as scheduled for removal |

## Multiple entries per PR

You can add more than one fragment per PR — for example a feature
addition that also fixes a related bug:

```bash
712.added.md   # New feature
712.fixed.md   # Bug fix in same PR
```

## Skipping the changelog check

The `Check Changelog` workflow enforces that every PR has at least one
fragment. If a PR has no user-facing impact (CI tweaks, internal
refactors), a maintainer can apply the `no changelog needed` label to
bypass the check.
