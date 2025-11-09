# Spring Boot application
FROM openjdk:17-jdk-slim

# Set working directory
WORKDIR /app

# Copy gradle files
COPY build.gradle settings.gradle gradlew ./
COPY gradle/ gradle/

# Copy source code
COPY src/ src/

# Install latest Gradle and build
RUN apt-get update && \
    apt-get install -y wget unzip && \
    wget https://services.gradle.org/distributions/gradle-8.5-bin.zip && \
    unzip gradle-8.5-bin.zip && \
    /app/gradle-8.5/bin/gradle build -x test

# Expose port
EXPOSE 8080

# Run the application
CMD ["sh", "-c", "find build/libs/ -name '*.jar' -exec java -jar {} \\;"]
