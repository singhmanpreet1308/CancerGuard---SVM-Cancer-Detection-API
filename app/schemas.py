from pydantic import BaseModel, Field


class CancerInput(BaseModel):
    """
    Input schema containing the 30 features expected
    by the trained CancerGuard SVM pipeline.
    """

    # Mean features
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float

    # Standard error features
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float

    # Worst features
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float


class CancerOutput(BaseModel):
    """
    Response returned by the prediction endpoint.
    """

    prediction: str = Field(
        ...,
        description="Cancer prediction: present or absent"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence between 0 and 1"
    )