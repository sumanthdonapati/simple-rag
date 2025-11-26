import os
import requests
import re as r
import torch

class UtilsError(Exception):
    def __init__(self, message, request_id=None):
        super().__init__(message)
        self.message = message
        self.request_id = request_id

def get_instance_name():
    """
    Get the instance name from Google Compute Engine metadata.
    """
    try:
        url = "http://metadata.google.internal/computeMetadata/v1/instance/id"
        headers = {"Metadata-Flavor": "Google"}
        response = requests.get(url, headers=headers, timeout=2)
        response.raise_for_status()
        return response.text
    except:
        return "Unknown Instance"

def getIP():
    try:
        d = str(urlopen('http://checkip.dyndns.com/', timeout=2).read())
        return r.compile(r'Address: (\d+\.\d+\.\d+\.\d+)').search(d).group(1)
    except:
        return "Unknown IP"

def get_public_ip_address():
    """
    Get the public IP address of the machine.
    """
    try:
        response = requests.get('https://api.ipify.org', timeout=2)
        response.raise_for_status()
        return response.text
    except:
        return "Unknown IP"

from urllib.request import urlopen

# GPU Metadata Initialization
gpu_instance_type = os.environ.get("gpu_instance_type","SECURE_CLOUD")

try:
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID","ENDPOINT_ID")
except:
    endpoint_id = "ENDPOINT_ID"


gpu_id = os.environ.get("RUNPOD_POD_ID","Runpod")
gpu_provider = "Runpod"

try:
    gpu_instance = torch.cuda.get_device_name()
except Exception:
    gpu_instance = "Unknown GPU"
    
try:
    gpu_ip_address = getIP()
except:
    gpu_ip_address = get_public_ip_address()

gpu_metrics = {
    "gpu_instance": gpu_instance, 
    "gpu_ip_address": gpu_ip_address,
    "gpu_id": gpu_id, 
    "gpu_cloud_provider": gpu_provider,
    "gpu_instance_type": gpu_instance_type 
}

def get_error_response(msg, code, type="serverless"):
    # Simplified for serverless only, removed Flask dependency
    return [{"error": msg, "status_code": code}, {"x-gpu-info": gpu_metrics}]
