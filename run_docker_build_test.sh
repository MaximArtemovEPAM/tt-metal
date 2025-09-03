#!/usr/bin/env bash

set -o xtrace
set -o nounset
set -o errexit

if docker images -q docker-conan-test > /dev/null 2>&1; then
    docker rmi docker-conan-test-tmp
fi

SOURCE_REPO="../tt-metal"
TARGET_DIR="../tt-metal-build-test"
WORKSPACE=$(pwd)

if [ ! -d "$TARGET_DIR" ]; then
    echo "Target directory '$TARGET_DIR' does not exist. Cloning from '$SOURCE_REPO'..."

    # Check if source repo exists
    if [ ! -d "$SOURCE_REPO" ]; then
        echo "Error: Source repository '$SOURCE_REPO' not found!"
        exit 1
    fi

    # Clone the repository
    git clone "$SOURCE_REPO" "$TARGET_DIR"
    echo "Successfully cloned '$SOURCE_REPO' to '$TARGET_DIR'"

else
    echo "Target directory '$TARGET_DIR' already exists. Updating with latest changes..."

    # Navigate to target directory
    cd "$TARGET_DIR"

    # Check if it's a git repository
    if [ ! -d ".git" ]; then
        echo "Error: '$TARGET_DIR' exists but is not a git repository!"
        exit 1
    fi

    # Get current branch name
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    echo "Current branch: $CURRENT_BRANCH"

    # Stash any local changes in target
    if ! git diff --quiet || ! git diff --staged --quiet; then
        echo "Stashing local changes in target..."
        git stash push -m "Auto-stash before update $(date)"
    fi

    # Fetch latest changes from source
    echo "Fetching latest changes..."
    git fetch origin

    # Pull latest changes
    echo "Pulling latest changes for branch '$CURRENT_BRANCH'..."
    git pull origin "$CURRENT_BRANCH"

    # Go back to original directory to check source repo
    cd $WORKSPACE

    # Check if source repo has unstaged changes and copy them
    cd "$SOURCE_REPO"
fi

docker build --progress=plain -f dockerfile/Dockerfile.basic-dev -t docker-conan-test-tmp . > ../docker_build_stdout.log 2>../docker_build_stderr.log
docker run --rm --mount type=bind,source=$(realpath $TARGET_DIR),target=/workspace docker-conan-test-tmp /workspace/run_docker_build_in_container.sh \
  > ../run_docker_result_stdout.log 2> ../run_docker_result_stderr.log
