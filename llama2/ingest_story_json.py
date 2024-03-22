import json
import os

def ingest_story_json(story_json_path: str) :
    with open(story_json_path, 'r') as f:
        story_json = json.load(f)

    return story_json





