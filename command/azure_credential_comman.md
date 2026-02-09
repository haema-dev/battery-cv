### uv env 설정

```bash
1. Dockerfile / pyproject.toml 작성
2. uv sync 명령어로 uv.lock 생성
```

### Azure Credential 생성

```bash
az ad sp create-for-rbac --name "[이름]" \
                         --role contributor \
                         --scopes /subscriptions/[구독ID]/resourceGroups/[리소스그룹명] \
                         --sdk-auth
```
