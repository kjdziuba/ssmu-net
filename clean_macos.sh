#!/bin/bash
# Quick cleanup script for macOS files
find . -name "._*" -type f -delete
find . -name ".DS_Store" -type f -delete
echo "Removed macOS metadata files"
