# Dockerfile best practices

A Dockerfile is a text file that contains instructions for building a Docker image.

## Simple example

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Key instructions

- `FROM` – base image
- `WORKDIR` – sets the working directory
- `COPY` / `ADD` – copy files into the image (prefer COPY)
- `RUN` – execute commands during build
- `ENV` – set environment variables
- `EXPOSE` – document the port the container listens on
- `CMD` / `ENTRYPOINT` – default command

## Layer caching tips

1. Order instructions from least to most frequently changing.
2. Copy dependency files (requirements.txt, package.json) before the full source code.
3. Use multi-stage builds to keep final images small.

## Multi-stage build example

```dockerfile
FROM golang:1.22 AS builder
WORKDIR /src
COPY . .
RUN CGO_ENABLED=0 go build -o /app

FROM alpine:3.20
COPY --from=builder /app /app
ENTRYPOINT ["/app"]
```
