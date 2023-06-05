#!/bin/bash

# Get the AWS account ID
aws_account_id=$(aws sts get-caller-identity --query Account --output text)
aws_region=$(aws configure get region)

echo "AccountId = ${aws_account_id}"
echo "Region = ${aws_region}"


# Create a new ECR repository
echo "Creating ECR Repository..."
aws ecr create-repository --repository-name rag-app

# Get the login command for the new repository
echo "Logging into the repository..."
#$(aws ecr get-login --no-include-email)
# aws ecr get-login-password --region ${aws_region} | docker login --username AWS --password-stdin ${aws_account_id}.dkr.ecr.${aws_region}.amazonaws.com

# Build and push the Docker image and tag it
echo "Building and pushing Docker image..."
sm-docker build -t "${aws_account_id}.dkr.ecr.us-east-1.amazonaws.com/rag-app:latest" --repository rag-app:latest .

