# FastADK RFC Process

## Introduction

The Request for Comments (RFC) process is designed to provide a consistent and controlled path for major changes to enter FastADK, allowing for proper design discussion and review before implementation begins.

## When to Use the RFC Process

You need to follow the RFC process if you're proposing a significant change to FastADK, including:

- Adding, removing, or modifying a major feature
- Making architectural changes that affect multiple components
- Changing public APIs in a way that affects backward compatibility
- Introducing new dependencies or technologies to the core framework

Small changes, bug fixes, documentation improvements, and minor enhancements can be handled through normal pull requests without an RFC.

## RFC Process Overview

1. **Idea Formation**: Discuss initial ideas in GitHub Discussions or Discord
2. **Draft RFC**: Create a new RFC document following the template
3. **Public Review**: Open a pull request with the RFC for community review
4. **Revision**: Incorporate feedback and revise as needed
5. **Decision**: Maintainers accept, reject, or request further changes
6. **Implementation**: Upon acceptance, implementation work can begin

## Creating an RFC

1. Fork the FastADK repository
2. Copy the RFC template from `.github/RFC_TEMPLATE.md`
3. Create a new file under `rfcs/` with the name `NNNN-descriptive-title.md` where `NNNN` is the next available number
4. Fill in the template with your proposal details
5. Open a pull request with your RFC

## Review Process

The review period will last at least 10 calendar days. During this time:

- Community members can review and comment on the RFC
- The author should address questions and concerns
- The RFC may be revised based on feedback
- Maintainers will evaluate the proposal's technical merit and alignment with project goals

## Decision Making

After the review period, maintainers will make one of the following decisions:

- **Accept**: The RFC is approved for implementation
- **Reject**: The RFC is declined with an explanation
- **Request for Changes**: The RFC requires specific modifications before it can be accepted
- **Postpone**: The RFC has merit but is not a current priority

## Implementation

Once an RFC is accepted:

1. The corresponding issue will be created for tracking implementation
2. The author or other contributors can begin working on implementation
3. Code changes will go through the normal PR review process
4. The RFC will be updated with implementation details and any deviations from the original plan

## RFC Maintenance

Accepted RFCs will be maintained as a historical record of design decisions. If implementation significantly deviates from the RFC, it should be updated to reflect the actual implementation.
