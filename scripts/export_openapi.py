import os

import yaml
from app.main import create_app

OUT_PATH = os.path.join("openapi", "openapi.yaml")


def main() -> None:
    app = create_app()
    spec = app.openapi()

    os.makedirs("openapi", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False, allow_unicode=True)

    print(f"Exported: {OUT_PATH}")


if __name__ == "__main__":
    main()
