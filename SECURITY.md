# Security Policy

## Reporting a vulnerability

Please do not report suspected vulnerabilities in a public issue.

Email `allisowang@apache.org` with the subject `pyspark-udtf security report`. Include a description, reproduction steps, affected versions, and any suggested remediation.

You should receive an acknowledgment within seven days. Please allow time to investigate and prepare a fix before publicly disclosing the issue.

## Supported versions

Security fixes are applied to the latest released version. Users should upgrade to the newest release before reporting an issue that may already be fixed.

## Credential safety

Several examples communicate with external services. Never commit API tokens, service credentials, private endpoints, or production data. Use environment variables or a secrets manager and provide only redacted values in reports and logs.
