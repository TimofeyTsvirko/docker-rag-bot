# Limiting container resources

Docker allows you to constrain the CPU and memory that a container can use.

## Memory limits

```bash
# Limit memory to 512 MB
docker run -m 512m --memory-swap 512m nginx

# Same with --memory
docker run --memory=512m nginx
```

- `-m` / `--memory` – hard limit on RAM.
- `--memory-swap` – total memory + swap. Setting it equal to `--memory` disables swap.

## CPU limits

```bash
# Use at most 50% of one CPU
docker run --cpus="0.5" nginx

# Pin to specific CPUs
docker run --cpuset-cpus="0,1" nginx
```

## In docker-compose.yml

```yaml
services:
  web:
    image: nginx
    deploy:
      resources:
        limits:
          cpus: "0.50"
          memory: 512M
        reservations:
          memory: 256M
```

(Note: the `deploy` key is mainly used by Swarm; for plain Compose some versions still honour the limits.)

## Why limit resources?

Without limits a single container can consume all host memory and cause the kernel OOM killer to terminate other processes.
