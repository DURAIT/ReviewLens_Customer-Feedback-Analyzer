"""Typed contracts for everything the LLM returns.

Allowed values live in ``Literal`` types, so an out-of-list value (for example a
category that is not allowed) fails validation and triggers the retry in
``chains.with_validation_retry``.
"""
from typing import Literal, get_args

from pydantic import BaseModel, Field, field_validator

Sentiment = Literal["positive", "neutral", "negative"]
Category = Literal[
    "food_quality", "delivery", "price", "staff", "packaging", "app", "other"
]
Urgency = Literal["low", "medium", "high"]
Language = Literal["English", "Tamil", "Tanglish", "Other"]

SENTIMENTS: tuple[str, ...] = get_args(Sentiment)
CATEGORIES: tuple[str, ...] = get_args(Category)
URGENCIES: tuple[str, ...] = get_args(Urgency)

CATEGORY_LABELS = {
    "food_quality": "Food / product quality",
    "delivery": "Delivery",
    "price": "Price and charges",
    "staff": "Staff",
    "packaging": "Packaging",
    "app": "App / website",
    "other": "Other",
}


def _not_blank(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must not be blank")
    return value


class ReviewAnalysis(BaseModel):
    """Output A: one analysed review. ``review_id`` is added by code, not the LLM."""

    sentiment: Sentiment = Field(description="Overall customer mood.")
    category: Category = Field(description="The single main topic of the review.")
    urgency: Urgency = Field(description="How fast the business should reply.")
    language: Language = Field(description="Language the review was written in.")
    summary: str = Field(description="One factual English sentence.")
    suggested_reply: str = Field(description="Polite, ready-to-post English reply.")

    @field_validator("summary", "suggested_reply")
    @classmethod
    def _strip(cls, value: str) -> str:
        return _not_blank(value)


class ThemeNote(BaseModel):
    """LLM-written insight for one category. Counts are added by pandas, never the LLM."""

    category: Category
    insight: str = Field(description="One sentence, max 25 words, no counts or percentages.")

    @field_validator("insight")
    @classmethod
    def _strip(cls, value: str) -> str:
        return _not_blank(value)


class WeeklyInsights(BaseModel):
    """Output B (narrative part): the numbers around it are computed in pandas."""

    complaints: list[ThemeNote]
    praises: list[ThemeNote]
    action_items: list[str] = Field(description="Exactly 3 concrete actions.")

    @field_validator("action_items")
    @classmethod
    def _items(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError("at least one action item is required")
        return cleaned[:3]
