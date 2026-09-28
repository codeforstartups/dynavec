import urllib.request
import json
import sys

def run():
    url = 'https://api.github.com/repos/ramashishmaurya/dynavec/actions/runs'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        runs = json.loads(response.read().decode())
    
    run_info = runs['workflow_runs'][0]
    
    jobs_url = run_info['jobs_url']
    req = urllib.request.Request(jobs_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        jobs = json.loads(response.read().decode())['jobs']
        
    for job in jobs:
        if job['conclusion'] == 'failure' and 'typecheck' in job['name'].lower():
            print(f"\n--- FAILED JOB: {job['name']} ---")
            log_url = f"https://api.github.com/repos/ramashishmaurya/dynavec/actions/jobs/{job['id']}/logs"
            try:
                req = urllib.request.Request(log_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as log_resp:
                    log_text = log_resp.read().decode('utf-8')
                    # Print the last 50 lines of the log
                    print("\n".join(log_text.splitlines()[-50:]))
            except Exception as e:
                print("Could not fetch log:", e)

if __name__ == '__main__':
    run()
