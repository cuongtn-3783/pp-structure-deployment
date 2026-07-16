"""Build-time only: instantiate PPStructureV3 on CPU to cache model weights.

Run during `docker build`, where no GPU is available. Instantiating the pipeline
triggers the one-time model download into an image layer so container startup and
the first request perform no network fetch.
"""

from pp_structure_deployment.pipeline import init_pipeline


def main() -> None:
    init_pipeline(device="cpu")
    print("PPStructureV3 weights cached")


if __name__ == "__main__":
    main()
