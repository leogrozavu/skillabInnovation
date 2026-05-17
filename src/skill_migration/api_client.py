from .config import SKILLAB_BASE_URL


class SkillabClient:
    def __init__(self, base_url=SKILLAB_BASE_URL, token=None, timeout=30):
        self.base_url = base_url.rstrip("/").removesuffix("/api")
        self.token = token
        self.timeout = timeout

    @property
    def headers(self):
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    def login(self, username, password):
        import requests

        response = requests.post(
            f"{self.base_url}/api/login",
            json={"username": username, "password": password},
            timeout=self.timeout,
        )
        response.raise_for_status()
        self.token = response.json()
        return self.token

    def post_form(self, path, data=None, params=None):
        import requests

        response = requests.post(
            f"{self.base_url}{path}",
            data=data or {},
            params=params or {},
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get(self, path, params=None):
        import requests

        response = requests.get(
            f"{self.base_url}{path}",
            params=params or {},
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_skills(self, keywords=None, page=1, page_size=20):
        data = {"keywords_logic": "or"}
        if keywords:
            data["keywords"] = keywords
        return self.post_form("/api/skills", data=data, params={"page": page, "page_size": page_size})

    def get_occupations(self, keywords=None, page=1, page_size=20):
        data = {"keywords_logic": "or"}
        if keywords:
            data["keywords"] = keywords
        return self.post_form(
            "/api/occupations",
            data=data,
            params={"page": page, "page_size": page_size},
        )

    def get_job_skill_demand(self, limit=100, source=None, from_date=None, to_date=None):
        params = {"limit": limit}
        if source:
            params["source"] = source
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date
        return self.get("/api/descriptive-analytics/jobs", params=params)
