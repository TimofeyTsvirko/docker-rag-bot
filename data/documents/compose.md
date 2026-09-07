# Docker Compose

Docker Compose is a tool for defining and running multi-container Docker applications.  
You use a YAML file to configure your application’s services, networks and volumes.

## Basic docker-compose.yml

```yaml
version: "3.8"
services:
  web:
    image: nginx:latest
    ports:
      - "8080:80"
    volumes:
      - ./html:/usr/share/nginx/html
  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: example
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

## Common commands

```bash
# Start in detached mode
docker compose up -d

# View logs
docker compose logs -f

# Stop and remove containers
docker compose down

# Rebuild images
docker compose build

# Scale a service
docker compose up -d --scale web=3
```

## Difference between `docker run` and Compose

- `docker run` starts a single container with many flags.
- Compose describes the whole stack declaratively and manages networking, volumes and dependencies automatically.

Compose is ideal for local development and testing of multi-service applications.
