FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files first for better caching
COPY package*.json ./

# Install dependencies with progress reporting
RUN npm install --progress=true

# Copy public directory with static assets
COPY public/ public/

# Copy source code
COPY src/ src/

# Start the application
CMD ["npm", "start"]
