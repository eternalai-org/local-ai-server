import requests

def health_check(url: str, timeout: int = 10) -> bool:
    """Wait for a service to be available at the given URL."""
    try:
        healthy_url = f"{url}/health"
        response = requests.get(healthy_url, timeout=timeout)
        return response.status_code == 200 and response.json()["status"] == "ok"
    except requests.exceptions.RequestException:
        return False