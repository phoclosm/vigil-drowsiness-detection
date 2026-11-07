"""Manual/CI real Tasks lifecycle smoke; requires a supplied external asset."""

import argparse

import numpy as np

from vigil.runtime.tasks_landmarks import TasksFaceLandmarkDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    with TasksFaceLandmarkDetector(args.model) as detector:
        assert detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)) == ()
    print("Real Tasks blank-frame lifecycle check passed; no face detection claimed.")


if __name__ == "__main__":
    main()
