## enviroMS Contributing Guidelines

A work in progress.

#### Steps to create a new version

1. Protect feature branch.
2. Set up Github mirroring (work with Yuri).
3. The next push to feature branch will trigger Github Actions to run on that branch. Ensure those checks are passing.
4. Bump enviroMS version using Makefile command (see semantic versioning guidelines).
5. Merge feature branch into main on Gitlab.
6. Use Github Actions to manually push the new Docker image to Dockerhub.
7. Ensure Github checks now pass on main, using the latest Dockerhub image.
8. Create a new release tag on Github.