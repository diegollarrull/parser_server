#!/bin/bash
uvicorn parser_server:app --port 9123 --workers 8

