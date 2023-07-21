#!/bin/bash

ngrok config add-authtoken 2Se5BBbG60l1AZySCCzcQGfY5Ox_5TybJvWeWkkTMfytKoZbb
nohup python main.py &
nohup ngrok http --domain safely-expert-lobster.ngrok-free.app 30007 &