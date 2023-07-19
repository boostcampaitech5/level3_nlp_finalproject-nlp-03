#!/bin/bash

# uvicorn app.main:app --port 8800 --host "0.0.0.0" --reload &
nohup python main.py &
nohup ngrok http --domain safely-expert-lobster.ngrok-free.app 30007 &