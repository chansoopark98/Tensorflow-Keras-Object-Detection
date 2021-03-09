import json
import sys

json_list = []

total_predictions = {
    'image_id': 111351,
    'category_id': 0,
    'bbox': [300, 200, 600, 800],
    'score': 0.9
}

dumps = json.dumps(total_predictions)
with open('test_json.json', 'w') as f:
    json.dump(total_predictions, f, indent=4)
