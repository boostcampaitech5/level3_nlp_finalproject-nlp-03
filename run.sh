#!/bin/bash

uvicorn app.main:app --port 8800 --host "0.0.0.0" --reload &
uvicorn modelapi.main:app --port 30007 --host "0.0.0.0" --reload &
# ngrok http --domain safely-expert-lobster.ngrok-free.app 30007 &