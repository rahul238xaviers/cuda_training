# ==============================================================================
# Containerized Environment for CUDA & C++ Systems Mastery Lab
# Base: NVIDIA CUDA 12.4 Devel (Ubuntu 22.04) + Clang + Node.js 20 LTS
# ==============================================================================

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Prevent interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# 1. Install Essential Toolchains: Clang, Build-Essential, Git, Curl, Clang-Format
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    clang-format \
    llvm \
    git \
    curl \
    ca-certificates \
    python3 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Node.js 20 LTS
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g npm@latest \
    && rm -rf /var/lib/apt/lists/*

# 3. Setup Workspace
WORKDIR /workspace

# 4. Copy dependency manifests first for Docker caching
COPY book/package.json book/package-lock.json* /workspace/book/

# Install Next.js dependencies
RUN cd /workspace/book && npm install

# 5. Copy the entire repository into container
COPY . /workspace

# 6. Expose Next.js Web Application Port
EXPOSE 3000

# 7. Environment Variables
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"
ENV NODE_ENV=development

# 8. Start Development Server
CMD ["npm", "--prefix", "book", "run", "dev"]
