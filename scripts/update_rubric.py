#!/usr/bin/env python3
"""Helper script to update research_rubric.json item statuses."""
import json
import sys
from datetime import datetime, timezone

def update_item(item_id, status, notes=None, error=None):
    with open('/home/codex/work/repo/research_rubric.json', 'r') as f:
        rubric = json.load(f)

    # Find and update the item
    for phase in rubric['phases']:
        for item in phase['items']:
            if item['id'] == item_id:
                item['status'] = status
                if notes is not None:
                    item['notes'] = notes
                if error is not None:
                    item['error'] = error
                break

    # Update summary counts
    counts = {'completed': 0, 'in_progress': 0, 'failed': 0, 'pending': 0}
    for phase in rubric['phases']:
        for item in phase['items']:
            counts[item['status']] = counts.get(item['status'], 0) + 1
    rubric['summary'] = {
        'total_items': 28,
        'completed': counts.get('completed', 0),
        'in_progress': counts.get('in_progress', 0),
        'failed': counts.get('failed', 0),
        'pending': counts.get('pending', 0),
    }
    rubric['updated_at'] = datetime.now(timezone.utc).isoformat()

    with open('/home/codex/work/repo/research_rubric.json', 'w') as f:
        json.dump(rubric, f, indent=2)
    print(f"Updated {item_id} -> {status}")

if __name__ == '__main__':
    item_id = sys.argv[1]
    status = sys.argv[2]
    notes = sys.argv[3] if len(sys.argv) > 3 else None
    error = sys.argv[4] if len(sys.argv) > 4 else None
    update_item(item_id, status, notes, error)
