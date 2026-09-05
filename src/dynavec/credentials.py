"""AWS credential handling — connect to the user's account many ways.

Supports (in order of precedence when multiple are given):
  * explicit access key / secret / optional session token
  * a named profile from ~/.aws/credentials
  * assume-role via STS (cross-account, with optional external id)
  * the default boto3 chain (env vars, instance role, SSO) when nothing is set

The resulting object is a frozen, side-effect-free description; call
:meth:`session` to materialize a ``boto3.Session``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AWSCredentials:
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None
    profile_name: Optional[str] = None
    region: Optional[str] = None

    # cross-account assume-role
    assume_role_arn: Optional[str] = None
    role_session_name: str = "dynavec"
    external_id: Optional[str] = None

    def session(self):
        """Build a ``boto3.Session`` from this credential description."""
        import boto3

        if self.assume_role_arn:
            return self._assume_role_session(boto3)

        kwargs = {}
        if self.access_key_id and self.secret_access_key:
            kwargs["aws_access_key_id"] = self.access_key_id
            kwargs["aws_secret_access_key"] = self.secret_access_key
            if self.session_token:
                kwargs["aws_session_token"] = self.session_token
        if self.profile_name:
            kwargs["profile_name"] = self.profile_name
        if self.region:
            kwargs["region_name"] = self.region
        return boto3.Session(**kwargs)

    def _assume_role_session(self, boto3):
        # Base session used only to call STS.
        base_kwargs = {}
        if self.access_key_id and self.secret_access_key:
            base_kwargs["aws_access_key_id"] = self.access_key_id
            base_kwargs["aws_secret_access_key"] = self.secret_access_key
            if self.session_token:
                base_kwargs["aws_session_token"] = self.session_token
        if self.profile_name:
            base_kwargs["profile_name"] = self.profile_name
        if self.region:
            base_kwargs["region_name"] = self.region

        base = boto3.Session(**base_kwargs)
        sts = base.client("sts")

        assume_kwargs = {
            "RoleArn": self.assume_role_arn,
            "RoleSessionName": self.role_session_name,
        }
        if self.external_id:
            assume_kwargs["ExternalId"] = self.external_id
        creds = sts.assume_role(**assume_kwargs)["Credentials"]

        return boto3.Session(
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
            region_name=self.region,
        )


def resolve_session(credentials: Optional[AWSCredentials], boto_session):
    """Pick a boto3 session: explicit session > credentials > default chain."""
    if boto_session is not None:
        return boto_session
    if credentials is not None:
        return credentials.session()
    import boto3

    return boto3.Session()
