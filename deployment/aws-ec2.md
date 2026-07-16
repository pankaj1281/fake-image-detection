# AWS EC2 Deployment Guide

1. Launch Ubuntu EC2, open ports 22/80/443.
2. Install Docker and Docker Compose.
3. Clone repository and run `docker compose up -d --build`.
4. Configure Route53 + TLS (Nginx certbot).
5. Add CloudWatch logs + auto-restart policy.
