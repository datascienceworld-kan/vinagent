import requests
from pathlib import Path
from tomlkit import document, table, array, dumps

def get_latest_version(package_name):
    try:
        resp = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=5)
        if resp.status_code == 200:
            return resp.json()["info"]["version"]
    except Exception:
        pass
    return None

def read_requirements(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.strip().startswith("#")
        ]

# Config
project_name = "vinagent"
author_name = "Tathao Nguyen"
author_email = "tathaonguyen@gmail.com"
top_level_packages = ["agent", "graph", "cache", "memory", "mcp", "config", "mlflow", "register"]

# Paths
req_main = Path("requirements.txt")
req_dev = Path("requirements-dev.txt")
pyproject_file = Path("pyproject.toml")
readme_file = Path("README.md")

# Read dependencies
main_deps = read_requirements(req_main)
dev_deps = read_requirements(req_dev)

# Build pyproject.toml
doc = document()

# [build-system]
build = table()
build.add("requires", ["setuptools>=61.0"])
build.add("build-backend", "setuptools.build_meta")
doc.add("build-system", build)

# [project]
project = table()
project.add("name", project_name)
project.add("version", "0.1.0")
project.add("description", "VinAgent multi-agent system")
project.add("requires-python", ">=3.8")
project.add("authors", [{"name": author_name, "email": author_email}])

# Only add README if it exists
if readme_file.exists():
    project.add("readme", "README.md")

# [project.dependencies]
deps_array = array()
for dep in main_deps:
    version = get_latest_version(dep)
    if version:
        deps_array.append(f"{dep}>={version}")
    else:
        deps_array.append(dep)
project.add("dependencies", deps_array)
doc.add("project", project)

# [project.optional-dependencies.dev]
if dev_deps:
    opt_deps = table()
    dev_array = array()
    for dep in dev_deps:
        version = get_latest_version(dep)
        if version:
            dev_array.append(f"{dep}>={version}")
        else:
            dev_array.append(dep)
    opt_deps.add("dev", dev_array)
    doc["project"]["optional-dependencies"] = opt_deps

# [tool.setuptools]
tool = table()
tool_setuptools = table()
tool_setuptools.add("packages", top_level_packages)
tool.add("setuptools", tool_setuptools)
doc.add("tool", tool)

# Write to pyproject.toml
with pyproject_file.open("w", encoding="utf-8") as f:
    f.write(dumps(doc))

print("✅ pyproject.toml đã được tạo đầy đủ (với tool.setuptools và check README).")
