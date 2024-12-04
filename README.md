# HCI - CritPred

This project serves as the central hub for all coding materials, documentation, and dependencies for the CritPred project.

## Repository Contents

### Summary of Branch Structure

This workflow promotes clean, manageable code and ensures a robust integration process. Protect `main` and `dev` branches to prevent direct pushes. The `dev` branch serves as the integration branch for feature branches, and changes are merged to `main` only after review.

- **main**: Stable production branch.
- **dev**: Integration branch for all features.
- **feature/feature-name**: Short-lived branches for individual features.

### Workflow

1. For each new feature, create a separate branch based on `dev`.
2. Once the feature is complete, create a pull request from `feature/feature-name` to `dev`.
3. After code review and approval, merge it into `dev`.
4. After all features for a release are merged into `dev`, create a pull request from `dev` to `main`.

## Use Case

### Prerequisites

### Setup

- **Dependencies**

### Install

### Usage

### Deployment

### Overview

### Hardware

### Software

### Architecture

### Key Features

### Maintenance

## Contribution Statement
