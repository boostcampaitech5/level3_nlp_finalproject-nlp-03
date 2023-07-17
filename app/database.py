from sqlalchemy import create_engine 
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


# db접속주소, sqlite3 db파일, 프로젝트 루트디렉토리
SQLALCHEMY_DATABASE_URL = 'sqlite:///./app/app.db'

# 커넥션 풀 생성: db에 접속하는 객체를 일정 갯수만큼 만들어 놓고 돌려가며 사용
engine=create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={'check_same_thread':False}
)

# db에 접속하기 위해 필요한 클래스
# autocommit=False -> 데이터변경후 commit을 해야만 실제로 저장, 잘못저장했을때 rollback가능
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# db모델을 구성할 때 사용하는 클래스
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()