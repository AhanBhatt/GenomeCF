# Docker and containers

GenomeCF includes:

- `Dockerfile`
- `environment.yml`
- `apptainer.def`
- `envs/caduceus.yml`

Typical local build:

```bash
docker build -t genomecf:local .
```

If Docker is unavailable, the repository can still be reproduced locally with:

```bash
pip install -e .[benchmark,dev]
```

