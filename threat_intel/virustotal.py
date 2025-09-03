import base64
import requests 
from numpy import random
import time

VT_API_KEY = "YOUR API KEY"

def get_report(url):
    try:
        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip('=')
        request = requests.get(f'https://www.virustotal.com/api/v3/urls/{url_id}',headers={"x-apikey": VT_API_KEY}).json()
        return (vt_flatten_url_object(request))
    except requests.RequestException as e:
        return(f'ERROR FOUND: {str(e)}')

def vt_flatten_url_object(report):
    data = report.get("data", {})  # condense and formats report 
    attrs = data.get("attributes", {})

    return {
        "url_id": data.get("id"),
        "url": attrs.get("url"),
        "tld": attrs.get("tld"),
        "title": attrs.get("title", ""),
        "times_submitted": attrs.get("times_submitted", 0),
        "reputation": attrs.get("reputation", 0),
        "votes": attrs.get("total_votes", {}),
        "last_analysis_date": attrs.get("last_analysis_date"),
        "first_submission_date": attrs.get("first_submission_date"),
        "last_modification_date": attrs.get("last_modification_date"),
        "last_analysis_stats": attrs.get("last_analysis_stats", {}),
        "last_final_url": attrs.get("last_final_url"),
        "engine_results": {
            eng: {
                "category": res.get("category"),
                "result": res.get("result")
            }
            for eng, res in attrs.get("last_analysis_results", {}).items()
        }
    }

    
