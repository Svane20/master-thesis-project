from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(...)


class InfoResponse(BaseModel):
    project_name: str = Field(..., alias="projectName")
    model_type: str = Field(..., alias="modelType")
    deployment_type: str = Field(..., alias="deploymentType")

    model_config = {
        "from_attributes": True,
        "populate_by_name": True
    }
