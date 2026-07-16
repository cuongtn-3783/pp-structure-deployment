"""Server entrypoint: launch uvicorn bound to 0.0.0.0:$PORT."""

import os

import uvicorn


def main() -> None:
    port = int(os.environ.get("PORT", "2603"))
    # workers=1 is mandatory: a single shared PPStructureV3 instance per process.
    uvicorn.run(
        "pp_structure_deployment.app:app",
        host="0.0.0.0",  # bind all interfaces inside the container
        port=port,
        workers=1,
    )


if __name__ == "__main__":
    main()
