from pydantic import BaseModel, Field


class AppealSchema(BaseModel):
    ttn_number: str = Field(
        ..., description="Валидный номер ТТН (например, 07xxxxxxxx или 8xxxxxxxx)"
    )
    problem_description: str = Field(..., description="Полное описание проблемы клиента")
