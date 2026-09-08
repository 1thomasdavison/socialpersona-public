from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import requests
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.cloud import storage

from .google_auth_env import build_google_credentials, ensure_google_auth_env_from_dotenv


TERMINAL_JOB_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PAUSED",
}


class VertexBatchInferenceClient:
    """Minimal REST client for Vertex Gemini batch inference with GCS input/output."""

    def __init__(
        self,
        *,
        project_id: str,
        location: str,
        model: str | None = None,
        timeout_seconds: int = 120,
        access_token_env: str = "VERTEX_ACCESS_TOKEN",
        request_retries: int = 5,
        retry_backoff_seconds: float = 2.0,
        max_retry_backoff_seconds: float = 30.0,
    ) -> None:
        self.project_id = (project_id or "").strip()
        self.location = (location or "").strip()
        if not self.project_id:
            raise ValueError("project_id is required")
        if not self.location:
            raise ValueError("location is required")
        self.model = self._normalize_model_path(model) if (model or "").strip() else ""
        self.timeout_seconds: float | None = float(timeout_seconds) if float(timeout_seconds) > 0 else None
        self.access_token_env = (access_token_env or "VERTEX_ACCESS_TOKEN").strip()
        self.request_retries = max(1, int(request_retries))
        self.retry_backoff_seconds = max(0.2, float(retry_backoff_seconds))
        self.max_retry_backoff_seconds = max(1.0, float(max_retry_backoff_seconds))
        self._gcs_client: storage.Client | None = None
        self._session = requests.Session()
        # Vertex API requests should not inherit shell-level network env overrides.
        self._session.trust_env = False
        self._adc_credentials: Any | None = None
        self._cached_access_token = ""
        self._cached_access_token_expiry_ts = 0.0
        self._ignore_env_access_token = False

    def upload_local_file(self, *, local_path: str | Path, gcs_uri: str) -> None:
        self._require_gcs_uri(gcs_uri, arg_name="gcs_uri")
        src = Path(local_path)
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"local_path not found: {src}")
        bucket_name, blob_name = self._split_gcs_uri(gcs_uri)
        client = self._get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(src))

    def submit_gcs_batch_job(
        self,
        *,
        input_gcs_uri: str,
        output_gcs_uri_prefix: str,
        display_name: str | None = None,
    ) -> dict[str, Any]:
        self._require_gcs_uri(input_gcs_uri, arg_name="input_gcs_uri")
        self._require_gcs_uri(output_gcs_uri_prefix, arg_name="output_gcs_uri_prefix")
        if not self.model:
            raise ValueError("model is required for submit_gcs_batch_job")

        payload: dict[str, Any] = {
            "displayName": (display_name or f"gemini-batch-{int(time.time())}").strip(),
            "model": self.model,
            "inputConfig": {
                "instancesFormat": "jsonl",
                "gcsSource": {
                    "uris": [input_gcs_uri],
                },
            },
            "outputConfig": {
                "predictionsFormat": "jsonl",
                "gcsDestination": {
                    "outputUriPrefix": output_gcs_uri_prefix,
                },
            },
        }

        return self._request_json(
            method="POST",
            url=self._batch_jobs_endpoint(),
            action="submit_gcs_batch_job",
            headers=self._auth_headers(),
            json=payload,
        )

    def get_batch_job(self, *, job_name_or_id: str) -> dict[str, Any]:
        job_name = self.to_job_name(job_name_or_id)
        return self._request_json(
            method="GET",
            url=f"{self._api_root()}/{job_name}",
            action="get_batch_job",
            headers=self._auth_headers(),
        )

    def wait_batch_job(
        self,
        *,
        job_name_or_id: str,
        poll_seconds: int = 30,
        max_wait_seconds: int = 3600,
    ) -> dict[str, Any]:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be > 0")

        job_name = self.to_job_name(job_name_or_id)
        deadline = (time.time() + max_wait_seconds) if max_wait_seconds > 0 else None
        last_state = ""
        while True:
            try:
                job = self.get_batch_job(job_name_or_id=job_name)
            except Exception as exc:
                now = time.time()
                if deadline is not None and now >= deadline:
                    raise TimeoutError(
                        f"Batch job polling failed until deadline for {job_name}: {exc}"
                    ) from exc
                sleep_for = min(float(poll_seconds), 30.0)
                print(
                    f"[vertex_batch] job={self.to_job_id(job_name)} poll_error={exc}; "
                    f"retrying in {sleep_for:.1f}s",
                    flush=True,
                )
                time.sleep(sleep_for)
                continue
            state = str(job.get("state", "")).strip()
            if state != last_state:
                print(f"[vertex_batch] job={self.to_job_id(job_name)} state={state}", flush=True)
                last_state = state
            if state in TERMINAL_JOB_STATES:
                return job
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError(
                    f"Batch job not finished within {max_wait_seconds}s: {job_name} (last state={state})"
                )
            time.sleep(poll_seconds)

    def download_output_prefix(
        self,
        *,
        output_gcs_uri_prefix: str,
        local_dir: str | Path,
        only_jsonl: bool = True,
    ) -> list[str]:
        self._require_gcs_uri(output_gcs_uri_prefix, arg_name="output_gcs_uri_prefix")
        target_dir = Path(local_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        all_uris = self.list_output_objects(output_gcs_uri_prefix=output_gcs_uri_prefix)
        if only_jsonl:
            uris = [uri for uri in all_uris if uri.endswith(".jsonl")]
            if not uris:
                uris = all_uris
        else:
            uris = all_uris

        downloaded: list[str] = []
        prefix_bucket, prefix_blob = self._split_gcs_uri(output_gcs_uri_prefix.rstrip("/") + "/")
        client = self._get_gcs_client()
        bucket = client.bucket(prefix_bucket)
        for uri in uris:
            bucket_name, blob_name = self._split_gcs_uri(uri)
            if bucket_name != prefix_bucket:
                raise RuntimeError(f"GCS bucket mismatch for uri={uri}, expected bucket={prefix_bucket}")

            relative_name = blob_name
            if prefix_blob and blob_name.startswith(prefix_blob):
                relative_name = blob_name[len(prefix_blob):].lstrip("/")
            if not relative_name:
                relative_name = Path(blob_name).name

            dest_path = target_dir / relative_name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(str(dest_path))
            downloaded.append(uri)
        return downloaded

    def list_output_objects(self, *, output_gcs_uri_prefix: str) -> list[str]:
        self._require_gcs_uri(output_gcs_uri_prefix, arg_name="output_gcs_uri_prefix")
        bucket_name, blob_prefix = self._split_gcs_uri(output_gcs_uri_prefix.rstrip("/") + "/")
        client = self._get_gcs_client()
        uris: list[str] = []
        for blob in client.list_blobs(bucket_name, prefix=blob_prefix):
            blob_name = str(blob.name or "").strip()
            if not blob_name or blob_name.endswith("/"):
                continue
            uris.append(f"gs://{bucket_name}/{blob_name}")
        return uris

    def get_output_uri_prefix(self, job: dict[str, Any]) -> str | None:
        output_cfg = job.get("outputConfig")
        if isinstance(output_cfg, dict):
            gcs_dest = output_cfg.get("gcsDestination")
            if isinstance(gcs_dest, dict):
                uri = str(gcs_dest.get("outputUriPrefix", "")).strip()
                if uri:
                    return uri

        # Some Vertex responses may also include outputInfo.gcsOutputDirectory.
        output_info = job.get("outputInfo")
        if isinstance(output_info, dict):
            uri = str(output_info.get("gcsOutputDirectory", "")).strip()
            if uri:
                return uri
        return None

    def to_job_name(self, job_name_or_id: str) -> str:
        raw = (job_name_or_id or "").strip()
        if not raw:
            raise ValueError("job_name_or_id is required")
        if raw.startswith("projects/"):
            return raw
        return f"projects/{self.project_id}/locations/{self.location}/batchPredictionJobs/{raw}"

    @staticmethod
    def to_job_id(job_name_or_id: str) -> str:
        raw = (job_name_or_id or "").strip().rstrip("/")
        if not raw:
            raise ValueError("job_name_or_id is required")
        return raw.split("/")[-1]

    @staticmethod
    def summarize_downloaded_jsonl(local_dir: str | Path) -> dict[str, Any]:
        root = Path(local_dir)
        jsonl_files = sorted(root.rglob("*.jsonl"))
        rows_total = 0
        rows_success = 0
        rows_failed = 0
        for path in jsonl_files:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                text = line.strip()
                if not text:
                    continue
                rows_total += 1
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    rows_failed += 1
                    continue
                status = str(row.get("status", "")).strip() if isinstance(row, dict) else ""
                if status:
                    rows_failed += 1
                else:
                    rows_success += 1

        return {
            "n_jsonl_files": len(jsonl_files),
            "n_rows_total": rows_total,
            "n_rows_success": rows_success,
            "n_rows_failed": rows_failed,
            "jsonl_files": [str(p) for p in jsonl_files],
        }

    def _api_root(self) -> str:
        endpoint_prefix = "" if self.location == "global" else f"{self.location}-"
        return f"https://{endpoint_prefix}aiplatform.googleapis.com/v1"

    def _batch_jobs_endpoint(self) -> str:
        return (
            f"{self._api_root()}/projects/{self.project_id}/locations/{self.location}/batchPredictionJobs"
        )

    def _auth_headers(self) -> dict[str, str]:
        access_token = self._resolve_access_token()
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    def _resolve_access_token(self) -> str:
        ensure_google_auth_env_from_dotenv()
        if not self._ignore_env_access_token:
            token = os.environ.get(self.access_token_env, "").strip()
            if token:
                return token

        now_ts = time.time()
        if self._cached_access_token and now_ts < (self._cached_access_token_expiry_ts - 60.0):
            return self._cached_access_token

        for attempt in range(1, 4):
            try:
                if self._adc_credentials is None:
                    self._adc_credentials, _ = build_google_credentials(
                        scopes=["https://www.googleapis.com/auth/cloud-platform"]
                    )
                credentials = self._adc_credentials
                if not credentials.valid or not str(getattr(credentials, "token", "") or "").strip():
                    credentials.refresh(GoogleAuthRequest())
                token = str(getattr(credentials, "token", "") or "").strip()
                if token:
                    expiry = getattr(credentials, "expiry", None)
                    expiry_ts = (
                        float(expiry.timestamp())
                        if expiry is not None and hasattr(expiry, "timestamp")
                        else (time.time() + 3000.0)
                    )
                    self._cached_access_token = token
                    self._cached_access_token_expiry_ts = expiry_ts
                    return token
            except Exception:
                if attempt < 3:
                    delay = min(self.retry_backoff_seconds * (2 ** (attempt - 1)), self.max_retry_backoff_seconds)
                    time.sleep(delay)

        try:
            token = subprocess.check_output(
                ["gcloud", "auth", "print-access-token"],
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(
                f"Missing access token: set env {self.access_token_env} "
                "or run `gcloud auth login` first."
            ) from exc
        if not token:
            raise ValueError(
                f"Missing access token: set env {self.access_token_env} "
                "or run `gcloud auth login` first."
            )
        return token

    def _request_json(
        self,
        *,
        method: str,
        url: str,
        action: str,
        headers: dict[str, str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, self.request_retries + 1):
            try:
                response = self._session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    timeout=self._request_timeout(),
                    **kwargs,
                )
                self._raise_for_status_with_body(response, action=action)
                text = str(response.text or "").strip()
                if not text:
                    return {}
                return response.json()
            except Exception as exc:
                last_exc = exc
                if self._is_access_token_expired_error(exc):
                    # If env token expired, switch to ADC/gcloud token refresh path.
                    self._ignore_env_access_token = True
                    self._cached_access_token = ""
                    self._cached_access_token_expiry_ts = 0.0
                if not self._is_retriable_exception(exc) or attempt >= self.request_retries:
                    raise
                delay = min(
                    self.retry_backoff_seconds * (2 ** (attempt - 1)),
                    self.max_retry_backoff_seconds,
                )
                print(
                    f"[vertex_batch] {action} attempt={attempt}/{self.request_retries} "
                    f"failed: {exc}; retrying in {delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"{action} failed without response")

    @staticmethod
    def _is_retriable_exception(exc: Exception) -> bool:
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(exc, requests.HTTPError):
            response = getattr(exc, "response", None)
            status = int(getattr(response, "status_code", 0) or 0)
            return status in {408, 409, 429, 500, 502, 503, 504}
        return False

    @staticmethod
    def _is_access_token_expired_error(exc: Exception) -> bool:
        if not isinstance(exc, requests.HTTPError):
            return False
        response = getattr(exc, "response", None)
        status = int(getattr(response, "status_code", 0) or 0)
        if status != 401:
            return False
        try:
            body = str(getattr(response, "text", "") or "")
        except Exception:
            body = ""
        lowered = body.lower()
        return "access_token_expired" in lowered or "unauthenticated" in lowered

    def _request_timeout(self) -> float | None:
        try:
            value = float(self.timeout_seconds)
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
        return value

    @staticmethod
    def _require_gcs_uri(uri: str, *, arg_name: str) -> None:
        if not str(uri).strip().startswith("gs://"):
            raise ValueError(f"{arg_name} must start with gs://, got: {uri}")

    @staticmethod
    def _normalize_model_path(model: str) -> str:
        cleaned = (model or "").strip()
        if not cleaned:
            raise ValueError("model is required")
        if cleaned.startswith(("publishers/", "projects/")):
            return cleaned
        if cleaned.startswith("models/"):
            return f"publishers/google/{cleaned}"
        if cleaned.startswith("google/"):
            suffix = cleaned.split("/", 1)[1].strip("/")
            return f"publishers/google/models/{suffix}"
        if "/models/" in cleaned:
            suffix = cleaned.split("/models/", 1)[1].strip("/")
            return f"publishers/google/models/{suffix}"
        return f"publishers/google/models/{cleaned}"

    def _get_gcs_client(self) -> storage.Client:
        if self._gcs_client is None:
            credentials, project_id = build_google_credentials(
                scopes=["https://www.googleapis.com/auth/devstorage.read_write"]
            )
            resolved_project = str(project_id or "").strip() or self.project_id
            self._gcs_client = storage.Client(project=resolved_project, credentials=credentials)
        return self._gcs_client

    @staticmethod
    def _raise_for_status_with_body(response: requests.Response, *, action: str) -> None:
        if response.ok:
            return
        body = ""
        try:
            body = (response.text or "").strip()
        except Exception:
            body = ""
        if len(body) > 1000:
            body = body[:1000] + "...(truncated)"
        raise requests.HTTPError(
            f"{action} failed: status={response.status_code}, body={body}",
            response=response,
        )

    @staticmethod
    def _split_gcs_uri(uri: str) -> tuple[str, str]:
        text = str(uri or "").strip()
        if not text.startswith("gs://"):
            raise ValueError(f"Invalid GCS uri: {uri}")
        tail = text[len("gs://"):]
        if "/" not in tail:
            return tail, ""
        bucket, blob = tail.split("/", 1)
        return bucket.strip(), blob.strip()
