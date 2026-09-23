from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime


DATABASE_URL = "sqlite:///./vaqelix.db"


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


Base = declarative_base()


class StudySession(Base):

    __tablename__ = "study_sessions"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    subject = Column(
        String(100),
        nullable=False
    )

    topic = Column(
        String(200),
        nullable=False
    )

    status = Column(
        String(50),
        default="Learning"
    )

    summary = Column(
        Text,
        default=""
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    messages = relationship(
        "Message",
        back_populates="study_session",
        cascade="all, delete-orphan"
    )


class Message(Base):

    __tablename__ = "messages"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    session_id = Column(
        Integer,
        ForeignKey("study_sessions.id"),
        nullable=False
    )

    role = Column(
        String(20),
        nullable=False
    )

    content = Column(
        Text,
        nullable=False
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    study_session = relationship(
        "StudySession",
        back_populates="messages"
    )


Base.metadata.create_all(
    bind=engine
)

