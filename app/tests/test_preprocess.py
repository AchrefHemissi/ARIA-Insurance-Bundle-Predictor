"""Unit tests for backend/app/services/preprocessor.py."""
import numpy as np
import pandas as pd
import pytest

from app.services.preprocessor import preprocess


class TestFrontendNormalization:
    """Verify that frontend-friendly field names / values are mapped to internal training names."""

    def test_annual_income_renamed(self, sample_df):
        result = preprocess(sample_df)
        assert "Estimated_Annual_Income" in result.columns or "log_Income" in result.columns

    def test_employment_status_mapped(self, sample_df):
        """Frontend sends 'Employed'; preprocessor should map to 'Employed_FullTime'."""
        result = preprocess(sample_df)
        assert result["Emp_Employed_FullTime"].iloc[0] == 1

    def test_deductible_tier_mapped(self, sample_df):
        """Frontend sends 'Medium'; preprocessor should map to 'Tier_2_Mid_Ded'."""
        result = preprocess(sample_df)
        assert result["Deductible_Tier_Ord"].iloc[0] == 2  # Tier_2_Mid_Ded → 2

    def test_month_int_to_string(self, sample_df):
        """Frontend sends month as int 3; preprocessor should resolve to 'March' then to numeric features."""
        result = preprocess(sample_df)
        assert result["Month_Num"].iloc[0] == 3


class TestFeatureEngineering:
    """Verify derived features are created correctly."""

    def test_total_dependents(self, sample_df):
        result = preprocess(sample_df)
        expected = sample_df["Adult_Dependents"].iloc[0] + sample_df["Child_Dependents"].iloc[0] + sample_df["Infant_Dependents"].iloc[0]
        assert result["Total_Dependents"].iloc[0] == expected

    def test_has_children(self, sample_df):
        result = preprocess(sample_df)
        assert result["Has_Children"].iloc[0] == 1  # Child_Dependents=2

    def test_fast_buyer(self, sample_df):
        result = preprocess(sample_df)
        assert result["Fast_Buyer"].iloc[0] == 1  # Days_Since_Quote=5 < 7

    def test_log_income_positive(self, sample_df):
        result = preprocess(sample_df)
        assert result["log_Income"].iloc[0] > 0

    def test_cyclical_month_features(self, sample_df):
        result = preprocess(sample_df)
        assert "Month_Sin" in result.columns
        assert "Month_Cos" in result.columns
        assert -1 <= result["Month_Sin"].iloc[0] <= 1
        assert -1 <= result["Month_Cos"].iloc[0] <= 1


class TestMissingValues:
    """Verify missing-value handling in preprocess."""

    def test_missing_child_dependents(self):
        row = {"User_ID": 1, "Age": 30, "Annual_Income": 40000, "Employment_Status": "Employed",
               "Adult_Dependents": 0, "Child_Dependents": None, "Infant_Dependents": 0,
               "Previous_Claims_Filed": 0, "Years_Without_Claims": 1, "Grace_Period_Extensions": 0,
               "Days_Since_Quote": 10, "Custom_Riders": 0, "Policy_Amendments_Count": 0,
               "Policy_Start_Month": 1, "Policy_Start_Week": 1, "Deductible_Tier": None,
               "Acquisition_Channel": None, "Existing_Policyholder": 0, "Num_Vehicles": 1}
        df = pd.DataFrame([row])
        result = preprocess(df)
        # Should not raise and fillna should produce valid output
        assert result["Deductible_Tier_Ord"].iloc[0] == 1  # defaults to Tier_1_High_Ded → 1
