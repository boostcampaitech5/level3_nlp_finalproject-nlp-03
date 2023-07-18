# 0. `cd app`

# 1. db생성
```python
alembic init migrations

#app/alembic.ini
sqlalchemy.url = sqlite:///./app.db

#app/migrations/env.py
import models
target_metadata = models.Base.metadata
```


# 2. table생성

```python
alembic revision --autogenerate
alembic upgrade head
```


# 3. test data 생성
```python
python testdata.py
```